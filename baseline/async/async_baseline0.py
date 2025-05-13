

#%%
import time
import sys
sys.path.append('/hackaton_medi')
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager      # stdlib ≥ 3.7
import pandas as pd
from mistralai import Mistral
from baseline.utils import ASSISTANT_INSTRUCTION
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest

# ──────────────────────────────────────────────────────────────────────────────
# helpers (unchanged – they are CPU‑only)
# ──────────────────────────────────────────────────────────────────────────────


def extract_medicine_name(user_query: str, list_medicine_name: list[str]):
    """
    Heuristic string‑match + fuzzy fallback.
    Returns *all* matches (caller chooses how to pick one).
    """
    matches = [
        name for name in list_medicine_name
        if name.lower() in user_query.lower()
    ]
    if not matches:
        from difflib import get_close_matches
        matches = get_close_matches(
            user_query.lower(),
            [n.lower() for n in list_medicine_name],
            cutoff=0.1,
        )
    return matches


# ──────────────────────────────────────────────────────────────────────────────
# main async baseline
# ──────────────────────────────────────────────────────────────────────────────
async def large_context_window_baseline_async(
        dict_q: dict,
        client: Mistral,
        model: str,
        df_fda_text: pd.DataFrame,
        test_mode: bool = True,
):
    """
    Async version – same return signature as the sync original:
        summary, recommendation_text, risk_score
    """
    tokenizer = MistralTokenizer.from_model(model)

    retry = True
    count = 0
    try:
        count += 1
        # 1) first call: extract drug + patient description
        query = dict_q["query"]
        first_resp = await client.chat.complete_async(
            model=model,
            messages=[
                {"role": "assistant", "content": ASSISTANT_INSTRUCTION},
                {"role": "user", "content": query},
            ],
            temperature=0,
            #top_k=0,  # Disable top-k sampling
            top_p=1.0,  # Disable nucleus sampling
            random_seed=42,
        )
        first_msg = first_resp.choices[0].message.content.strip()

        if first_msg == "I am not sure I understand the name of  drug, can you be more precise?":
            err = "Model could not extract the drug name from the query."
            raise ValueError(err) if test_mode else NotImplementedError(err)

        drug_name, description_patient = first_msg.split(maxsplit=1)
        print(f"drug name: {drug_name}")

        # 2) locate the canonical FDA sheet
        list_medicine_name = df_fda_text["name"].tolist()
        medicine_name = extract_medicine_name(drug_name, list_medicine_name)[0]
        description = df_fda_text.loc[
            df_fda_text["name"] == medicine_name.upper(), "fda_text"
        ].values[0]

        messages=[
            {
                "role": "assistant",
                "content": (
                    f"You are a helpful assistant to a doctor who wants to prescribe {medicine_name}.\n\n"
                    f"The doctor describes the patient as follows: {description_patient}\n\n"
                    f"Here is the drug information sheet for {medicine_name}: {description}\n\n"
                    f"First, summarise the risk information sheet.\n\n"
                    f"Then, based on the patient description and the drug information, structure your response as "
                    f"follows:\n\n"
                    f"### Recommendation\n"
                    f"Give your professional opinion about prescribing this drug to the patient.\n\n"
                    f"Finally, assign a risk score and present it exactly as:\n"
                    f"#### Risk score: either 0 or 1 or 2\n\n"
                    f"(0=appropriate, 1=uncertain/potentially dangerous, 2=inappropriate and dangerous).\n\n"
                    f"Do not add any extra commentary outside the required formats."
                ),
            },
            {"role": "user", "content": query},
        ]

        # 3) second call: answer with full context window
        final_resp = await client.chat.complete_async(
            model=model,
            temperature=0,
            #top_k=0,  # Disable top-k sampling
            top_p=1.0,  # Disable nucleus sampling
            random_seed=42,
            messages=messages,
        )
        tokens = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages))


        text = final_resp.choices[0].message.content
        summary, recommendation_block = text.split("### Recommendation", 1)
        risk_score = recommendation_block.split("#### Risk score:")[-1].strip()

        print(f"risk score lcw: {risk_score}")

        #return summary, recommendation_block, risk_score
        return {
            "drug_name": dict_q["drug_name"],
            "profile": dict_q["profile"],
            "justification": dict_q["gr_justification"],
            "summary": dict_q["gr_summary"],
            "risk_score": dict_q["gr_risk_score"],
            "lwc_summary": summary,
            "lwc_recommendation_text": recommendation_block,
            "lwc_Risk_score": risk_score,
            "profile_id": dict_q["profile_id"],
            'tokens': len(tokens.tokens),
            "words": len(tokens.text.split('▁')),
        }
    except Exception as e:
        print(f"Error in RAG baseline: {e}")
        if 'Requests rate limit exceeded' not in str(e):
            retry = False
        if not retry:
            return {
                "drug_name": dict_q["drug_name"],
                "profile": dict_q["profile"],
                "justification": dict_q["gr_justification"],
                "summary": dict_q["gr_summary"],
                "risk_score": dict_q["gr_risk_score"],
                "lwc_summary": None,
                "lwc_recommendation_text": None,
                "lwc_Risk_score": None,
                "profile_id": dict_q["profile_id"],
            }
        print(f"sleeping for 2 seconds... because of rate limit")
        time.sleep(2)
    if count > 5:
        retry = False
        print("Retry limit reached, stopping.")
        return None, None, None, None


# ──────────────────────────────────────────────────────────────────────────────
# optional helper: run a list of queries concurrently (rate‑limited)
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def acquire(semaphore: asyncio.Semaphore | None):
    if semaphore:
        await semaphore.acquire()
        try:
            yield
        finally:
            semaphore.release()
    else:
        yield


async def _one_lcw(
        dict_q: dict,
        client: Mistral,
        sem: asyncio.Semaphore | None,
        model: str,
        df_fda_text: pd.DataFrame,
):
    async with acquire(sem):
        return await large_context_window_baseline_async(
            dict_q=dict_q,
            client=client,
            model=model,
            df_fda_text=df_fda_text,
        )


async def run_many_lcw(
        list_input_async: list[dict],
        model: str,
        df_fda_text: pd.DataFrame,
        api_key: str,
        max_concurrency: int = 5,
):
    sem = asyncio.Semaphore(max_concurrency)

    async with Mistral(api_key=api_key) as client:
        coros = [_one_lcw(q, client, sem, model, df_fda_text) for q in list_input_async]
        # progress bar + gather (requires tqdm ≥ 4.66)
        from tqdm.asyncio import tqdm_asyncio
        results = await tqdm_asyncio.gather(*coros)
    return results

#%%
# ──────────────────────────────────────────────────────────────────────────────
# demo / CLI entry‑point
# ──────────────────────────────────────────────────────────────────────────────



if __name__ == "__main__":
    import os

    df_fda_text = pd.read_csv(
        "/hackaton_medi/fda_data/result/drugs_complete_rag.csv"
    )
    api_key = os.getenv("MISTRAL_API_KEY")

    model_name = "mistral-small-latest"

    demo_queries = [
        "Should my patient take Acyclovir knowing he has severe kidney failure?",
        "Should my patient take REGADENOSON medicine knowing she has severe cardiac failure and is on anti‑arrhythmic treatment?",
    ]

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

    results = asyncio.run(
        run_many_lcw(
            list_input_async=list_input_async,
            model=model_name,
            df_fda_text=df_fda_text,
            api_key=api_key,
            max_concurrency=3,
        )
    )

    for q, (summary, rec_block, risk) in zip(demo_queries, results):
        print("\n", "=" * 80)
        print("QUERY:", q)
        print("RISK SCORE:", risk)
