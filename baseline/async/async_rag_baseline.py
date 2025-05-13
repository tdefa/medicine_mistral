#%%
import sys

sys.path.append('/hackaton_medi')
import os
import asyncio
from pathlib import Path
import re
import faiss
import numpy as np
from mistralai import Mistral
import pandas as pd
from baseline.chunk import chunk_by_section
from baseline.utils import ASSISTANT_INSTRUCTION, ASSISTANT_INSTRUCTION_DECOMPOSITION
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest


def sync_get_text_embedding(input, client):
    embeddings_batch_response = client.embeddings.create(
        model="mistral-embed",
        inputs=input
    )
    return embeddings_batch_response.data[0].embedding


# ──────────────────────────────────────────────────────────────────────────────
# Async helpers
# ──────────────────────────────────────────────────────────────────────────────
async def get_text_embedding_async(texts, client, model="mistral-embed"):
    """
    Embed a single string or a list of strings with the async Mistral SDK.
    Always returns a list[np.ndarray] of shape (len(texts), dim).
    """
    # normalise input
    if isinstance(texts, str):
        texts = [texts]

    resp = await client.embeddings.create_async(model=model, inputs=texts)
    return [np.array(d.embedding, dtype=np.float32) for d in resp.data]

# ──────────────────────────────────────────────────────────────────────────────
# Main async RAG baseline
# ──────────────────────────────────────────────────────────────────────────────
async def rag_baseline_async(
        dict_q: str,
        client: Mistral,
        model: str,
        drugs_df,
        chunk_size: int = 60,
        test_mode: bool = True,
        path_databased: str = "/home/tom/Bureau/phd/mistral_training/hackaton_medi/rag_vector/IndexFlatL2",
        k_rag=4,
        model_rag="baseline",
):
    """
    Asynchronous version of rag_baseline.

    Identical return signature:
        summary, recommendation_text, risk_score_str, retrieved_chunk_list
    """

    tokenizer = MistralTokenizer.from_model(model)

    print()
    assert model_rag in ["baseline", "decomposition", "baseline_hybride"]
    retry = True
    count = 0
    query = dict_q["query"]
    print(f"Processing query: {query}")
    #print(f"drug_name: {dict_q}")
    while retry:
        try:
            count += 1
            # ── 1. Extract drug name & patient description ────────────────────────────

            if model_rag == "baseline" or model_rag == "baseline_hybride":
                assistant_instruction = ASSISTANT_INSTRUCTION

            elif model_rag == "decomposition":
                assistant_instruction = ASSISTANT_INSTRUCTION_DECOMPOSITION
            else:
                raise ValueError("Invalid model_rag. Choose either 'baseline' or 'decomposition or 'baseline_hybride'.")

            chat_resp = await client.chat.complete_async(
                model=model,
                messages=[
                    {"role": "assistant", "content": assistant_instruction},
                    {"role": "user", "content": query},
                ],
                temperature=0,
                #top_k=0,  # Disable top-k sampling
                top_p=1.0,  # Disable nucleus sampling
                random_seed=42,
            )
            first_reply = chat_resp.choices[0].message.content.strip()
            print(first_reply)

            if first_reply == "I am not sure I understand the name of  drug, can you be more precise?":
                raise ValueError("Model could not extract the drug name.") if test_mode else NotImplementedError

            medicine_name, description_patient = map(str.strip, first_reply.split(":", 1))
            if medicine_name[0]==  '"':
                medicine_name = medicine_name[1:]
            print(f"medicine name: {medicine_name}")

            # ── 2. Retrieve or build FAISS index for this medicine ────────────────────
            description = drugs_df.loc[drugs_df["name"] == medicine_name.upper(), "fda_text"].values[0]
            chunks = chunk_by_section(description, chunk_size)

            index_path = Path(path_databased) / f"{medicine_name.upper()}.index"
            if index_path.exists():
                print(f"Loading previous vector database for {medicine_name.upper()}")
                index = faiss.read_index(str(index_path))
            else:
                print("No index found, creating a new one.")
                # batch‑embed all chunks in one request
                chunk_embeddings = await get_text_embedding_async(chunks, client)
                emb_matrix = np.vstack(chunk_embeddings).astype("float32")

                dim = emb_matrix.shape[1]
                index = faiss.IndexFlatL2(dim)
                index.add(emb_matrix)
                faiss.write_index(index, str(index_path))

            if model_rag == "baseline" or model_rag == "baseline_hybride":
                # ── 3. Embed the patient description & search ────────────────────────────
                try:
                    question_emb = (await get_text_embedding_async(description_patient, client))[0][None, :]
                except Exception as e:
                    print(f"Error in embedding: {e}")
                    question_emb = sync_get_text_embedding(description_patient, client)
                D, I = index.search(question_emb, k=k_rag)
                retrieved_chunk = [chunks[i] for i in I[0]]

            elif model_rag == "decomposition":
                extracted_sentences = re.findall(r'\[(.*?)\]', description_patient)
                description_patient = [sentence.strip() for sentence in extracted_sentences] + [description_patient]
                print(f"extracted sentences: {description_patient}")
                retrieved_chunk = []
                for sentence in description_patient:
                    question_emb = (await get_text_embedding_async(sentence, client))[0][None, :]
                    D, I = index.search(question_emb, k=k_rag)
                    retrieved_chunk += [chunks[i] for i in I[0]]

            else:
                raise ValueError("Invalid model_rag. Choose either 'baseline' or 'decomposition' or 'baseline_hybride'.")

            if model_rag == "baseline_hybride":
                gr_summary = dict_q["gr_summary"]
                messages=[
                    {
                        "role": "assistant",
                        "content": (
                            f"You are a helpful assistant to a doctor who wants to prescribe {medicine_name} to a patient.\n\n"
                            f"The doctor describes the patient as follows: {description_patient}\n\n"
                            f"Here are are the summary of the drug information  {gr_summary}\n\n"
                            f"Here are extracts of the drug information sheet for {medicine_name}: {retrieved_chunk}\n\n"
                            f"First, summarise the information sheet.\n\n"
                            f"Then, based on the patient description and the drug information, structure your response as follows:\n\n"
                            f"### Recommendation\n"
                            f"Give your professional opinion about prescribing this drug to the patient.\n\n"
                            f"Finally, assign a risk score and present it exactly as:\n"
                            f"#### Risk score: either 0 or 1 or 2\n\n"
                            f"where 0 = appropriate, 1 = uncertain/potentially dangerous, 2 = inappropriate and dangerous.\n\n"
                            f"Do not add any extra commentary outside the required formats."
                        ),
                    },
                    {"role": "user", "content": query},
                ]

            else:
                messages=[
                    {
                        "role": "assistant",
                        "content": (
                            f"You are a helpful assistant to a doctor who wants to prescribe {medicine_name} to a patient.\n\n"
                            f"The doctor describes the patient as follows: {description_patient}\n\n"
                            f"Here are extracts of the drug information sheet for {medicine_name}: {retrieved_chunk}\n\n"
                            f"First, summarise the information sheet.\n\n"
                            f"Then, based on the patient description and the drug information, structure your response as follows:\n\n"
                            f"### Recommendation\n"
                            f"Give your professional opinion about prescribing this drug to the patient.\n\n"
                            f"Finally, assign a risk score and present it exactly as:\n"
                            f"#### Risk score: either 0 or 1 or 2\n\n"
                            f"where 0 = appropriate, 1 = uncertain/potentially dangerous, 2 = inappropriate and dangerous.\n\n"
                            f"Do not add any extra commentary outside the required formats."
                        ),
                    },
                    {"role": "user", "content": query},
                ]

            # ── 4. Final recommendation prompt ───────────────────────────────────────
            final_resp = await client.chat.complete_async(
                model=model,
                temperature=0,
                #top_k=0,  # Disable top-k sampling
                top_p=1.0,  # Disable nucleus sampling
                random_seed=42,
                messages=messages,
            )

            tokens = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages))


            response_text = final_resp.choices[0].message.content
            summary, recommendation_block = response_text.split("### Recommendation", 1)
            risk_score = recommendation_block.split("#### Risk score:")[-1].strip()

            print(f"Risk_score pred by rag: {risk_score}")
            return {
                "drug_name": dict_q["drug_name"],
                "profile": dict_q["profile"],
                "justification": dict_q["gr_justification"],
                "summary": dict_q["gr_summary"],
                "risk_score": dict_q["gr_risk_score"],
                "rag_summary": summary,
                "rag_recommendation_text": recommendation_block,
                "rag_Risk_score": risk_score,
                "profile_id": dict_q["profile_id"],
                "retrieved_chunk": retrieved_chunk,
                'tokens': len(tokens.tokens),
                "words": len(tokens.text.split('▁')),
            }
        except Exception as e:
            print(f"Error in RAG baseline: {e}")
            if 'Requests rate limit exceeded' not in str(e):
                print(f"error different than rate limit not retrying for {dict_q['drug_name']}")
                retry = False
            if not retry:
                return {
                    "drug_name": dict_q["drug_name"],
                    "profile": dict_q["profile"],
                    "justification": dict_q["gr_justification"],
                    "summary": dict_q["gr_summary"],
                    "risk_score": dict_q["gr_risk_score"],
                    "rag_summary": None,
                    "rag_recommendation_text": None,
                    "rag_Risk_score": None,
                    "profile_id": dict_q["profile_id"],
                    "retrieved_chunk": None
                }
            print(f"Sleeping for 2 seconds... because of rate limit")
            import time
            time.sleep(2)
        if count > 5:
            retry = False
            print("Retry limit reached, stopping.")
            return None, None, None, None

        """dict_res = {
            "drug_name": drug_name,
            "profile": profile,
            "justification": gr_justification,
            "summary": gr_summary,
            "risk_score": gr_risk_score,
            "rag_summary": rag_summary,
            "rag_recommendation_text": rag_recommendation_text,
            "rag_Risk_score": rag_Risk_score,
            "profile_id": profile_id,
            "retrieved_chunk": retrieved_chunk
        }"""


import asyncio
from contextlib import asynccontextmanager  # Python 3.11+
from mistralai import Mistral
from tqdm.asyncio import tqdm_asyncio  # progress bar for async loops


# ──────────────────────────────────────────────────────────────────────────────
# Small helper so we can "async with sem:" even on 3.10
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def acquire(sem: asyncio.Semaphore | None):
    if sem:
        await sem.acquire()
        try:
            yield
        finally:
            sem.release()
    else:
        yield


# ──────────────────────────────────────────────────────────────────────────────
# One task for one query (wrapped in a semaphore)
# ──────────────────────────────────────────────────────────────────────────────
async def _one_query(
        dict_q: dict,
        client: Mistral,
        sem: asyncio.Semaphore | None,
        model: str,
        drugs_df,
        k_rag: int = 4,
        model_rag: str = "baseline"  # or "decomposition"
):
    print(f"Processing query: {dict_q['query']}")
    async with acquire(sem):
        return await rag_baseline_async(
            dict_q=dict_q,
            client=client,
            model=model,
            drugs_df=drugs_df,
            k_rag=k_rag,
            model_rag=model_rag
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main fan‑out / fan‑in runner
# ──────────────────────────────────────────────────────────────────────────────
async def run_many_queries(
        list_input_async: list[dict],
        model: str,
        drugs_df,
        api_key: str,
        max_concurrency: int = 5,
        k_rag=4,  # number of chunks to retrieve
        # tweak to stay under your rate‑limit$
        model_rag: str = "baseline"  # or "decomposition"
):
    """
    Returns a list in the same order as `queries`, each item being
    (summary, recommendation_text, risk_score, retrieved_chunk).
    """
    sem = asyncio.Semaphore(max_concurrency)
    print(f"Max concurrency: {max_concurrency}")

    async with Mistral(api_key=api_key) as client:
        # create all the coroutines up‑front
        coros = [
            _one_query(q, client, sem, model, drugs_df,
                       k_rag=k_rag, model_rag=model_rag)
            for q in list_input_async
        ]

        # You can use asyncio.gather directly, or keep an async progress bar
        # If tqdm isn't needed, just: results = await asyncio.gather(*coros)
        #results = [
        #    await task
        #    async for task in tqdm_asyncio.as_completed(coros, total=len(coros))
        #]
        results = await tqdm_asyncio.gather(*coros)

    return results


#%%
# ──────────────────────────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    list_input_async = [{
        "drug_name": "Minocycline",
        "profile": " A 27-year-old female who is 28 weeks pregnant presents with a urinary tract infection",
        "gr_justification": None,
        "gr_summary": None,
        "gr_risk_score": 2,
        "profile_id": 213,
        "query": "Should my patient takes following drug 'Minocycline', knowing its profile :"
                 "  A 27-year-old female who is 28 weeks pregnant presents with a urinary"
                 " tract infection caused by Escherichia coli. She has no known allergies.?",
        "description": None
    }, {
        "drug_name": "Minocycline",
        "profile": " A 27-year-old female who is 28 weeks pregnant presents with a urinary tract infection",
        "gr_justification": None,
        "gr_summary": None,
        "gr_risk_score": 2,
        "profile_id": 213,
        "query": "Should my patient takes following drug 'Minocycline', knowing its profile :"
                 "  A 27-year-old female who is 28 weeks pregnant presents with a urinary"
                 " tract infection caused by Escherichia coli. She has no known allergies.?",
        "description": None
    }]
    drugs_df = pd.read_csv \
        ('/hackaton_medi/fda_data/result/drugs_complete_rag.csv')
    import time

    t = time.time()  # start timer
    all_results = asyncio.run(
        run_many_queries(
            list_input_async=list_input_async,
            model="mistral-small-latest",
            drugs_df=drugs_df,
            api_key=os.getenv("MISTRAL_API_KEY"),
            max_concurrency=2,  # ≤ 5 concurrent calls at any moment
        )
    )
    print("Elapsed time:", time.time() - t)

    # unpack or post‑process as you like
    for q, (summary, recommendation, risk, chunks) in zip(queries, all_results):
        print("=" * 80)
        print(q)
        print(summary[:200], "…")  # preview
        print("Risk score:", risk)
