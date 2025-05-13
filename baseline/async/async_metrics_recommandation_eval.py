"""
Async evaluation script: large‑vs‑small LLM recommendations
────────────────────────────────────────────────────────────
• Makes thousands of judge calls and embedding calls in parallel
• Uses a semaphore so you don’t exceed the provider’s rate limits
• Computes judge agreement and cosine similarities at the end
"""

import os
import asyncio
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from mistralai import Mistral

# ── CONFIG ───────────────────────────────────────────────
MODE        = "lwc"                       # "lwc" or "rag"
MODEL_JUDGE = "mistral-large-latest"
EMB_MODEL   = "mistral-embed"
API_KEY =     os.getenv("MISTRAL_API_KEY")
DF_FOLDER   = "/home/tom/Bureau/phd/mistral_training/hackaton_medi/result_baseline/lwc11mai/best_copy"
MAX_PARALLEL = 3                          # keep it polite :-)

# ── ASYNC HELPERS ────────────────────────────────────────
sem = asyncio.Semaphore(MAX_PARALLEL)
client = Mistral(api_key=API_KEY)

async def llm_judge(large_txt: str, small_txt: str) -> int:
    """Return 2 (match) / 1 (partial) / 0 (diff) from the judge."""
    system_msg = (
        "You are an expert clinical reviewer. Decide whether the TWO "
        "drug‑prescription recommendations below express the SAME clinical "
        "recomandation. Reply with one digit:\n"
        "2 – same information \n1 – similar but some information is not matched \n0 – different."
    )
    user_msg = (
        f"### Recommendation A \n{large_txt}\n\n"
        f"### Recommendatio B \n{small_txt}\n\n"
        "Do Recommendation A match Recommendation B ?  Reply with one digit:"
        " 2 – same information \n1 – similar but some information is not matched \n0 – different."
    )
    async with sem:
        resp = await client.chat.complete(
            model=MODEL_JUDGE,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=1,
            temperature=0,
        )
    print(resp.choices[0].message.content)
    try:
        return int(resp.choices[0].message.content.strip())
    except Exception:
        return None

async def get_embed(text: str) -> np.ndarray:
    """Return a single embedding vector."""
    async with sem:
        resp = await client.embeddings.create(model=EMB_MODEL, inputs=text)
    return np.array(resp.data[0].embedding)

async def pairwise_similarity(text_big: str, text_small: str) -> float:
    """Cosine similarity via async embeddings."""
    vec_big, vec_small = await asyncio.gather(
        get_embed(text_big),
        get_embed(text_small),
    )
    return float(cosine_similarity(vec_big.reshape(1, -1),
                                   vec_small.reshape(1, -1))[0, 0])

# ── MAIN ─────────────────────────────────────────────────
async def process_dataframe(df: pd.DataFrame) -> None:
    big_texts  = df["summary"].tolist()            # reference
    if MODE == "rag":
        small_texts = df["rag_recommendation_text"].tolist()
    else:
        small_texts = df["lwc_recommendation_text"].tolist()

    profile_names = df["profile"].tolist()

    # Fire off judge tasks
    judge_tasks = [llm_judge(b, s) for b, s in zip(big_texts, small_texts)]

# Fire off similarity tasks
    sim_tasks   = [pairwise_similarity(b, s) for b, s in zip(big_texts, small_texts)]


    judge_results, sim_results = await asyncio.gather(
        tqdm_asyncio.gather(*judge_tasks, desc="Judge Tasks"),
        tqdm_asyncio.gather(*sim_tasks, desc="Similarity Tasks"),
    )

    judge_acc = sum(judge_results) / (2 * len(judge_results))  # normalise (2=perfect)
    print(f"▶ Judge agreement (0‑2 scaled): {judge_acc:.3f}")

    mean_sim  = sum(sim_results) / len(sim_results)
    print(f"▶ Mean cosine similarity:       {mean_sim:.3f}")
    return {judge_results : "judge_results", sim_results : "sim_results",
            "profile_names" : profile_names}

def load_all_csv(folder: str) -> list[pd.DataFrame]:
    return [
        pd.read_csv(os.path.join(folder, f))
        for f in os.listdir(folder)
        if f.endswith(".csv")
    ]
from tqdm.asyncio import tqdm_asyncio        # progress bar for async loops

async def main(max_concurrency):

    sem = asyncio.Semaphore(max_concurrency)
    print(f"Max concurrency: {max_concurrency}")

    dfs = load_all_csv(DF_FOLDER)
    async with Mistral(api_key=API_KEY) as client:
        tasks = [process_dataframe(df) for df in dfs]
        results = await tqdm_asyncio.gather(*tasks)
        await asyncio.gather(*tasks)
    await client.aclose()               # tidy shutdown
    return results

if __name__ == "__main__":
    results = asyncio.run(main(MAX_PARALLEL))
