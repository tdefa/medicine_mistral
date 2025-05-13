

import os
import time

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from mistralai import Mistral
from tqdm import tqdm
# pip install mistralai
# from mistralai.async_client import MistralAsyncClient   # for async variant

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
MODEL_JUDGE = "mistral-large-latest"
API_KEY = os.getenv("MISTRAL_API_KEY") # put yours here or in env

# ────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────
client = Mistral(api_key=API_KEY)

def llm_judge_sync(large_text: str, small_text: str) -> int:
    """
    Returns 1 if the two recommendations are judged equivalent, else 0.
    """
    system_msg = (
        "You are an expert clinical reviewer. Your job is to decide whether the TWO "
        "drug‑prescription recommendations below convey the SAME clinical advice.\n\n"
        "Reply with **exactly one digit** (no other words or punctuation):\n"
        "2 – Clinically equivalent. All critical information (risk level, indication/contraindication, dosage advice) matches.\n"
        "1 – Generally similar, but some non‑critical or minor details differ.\n"
        "0 – Clinically different. Any critical aspect conflicts or is missing.\n"
    )

    user_msg = (
        f"### Recommendation A\n{large_text}\n\n"
        f"### Recommendation B\n{small_text}\n\n"
        "Do Recommendation A and Recommendation B convey the same clinical advice?\n"
        "Answer with a single digit: 2, 1, or 0."
    )
    resp = client.chat.complete(
        model=MODEL_JUDGE,
        messages=[
            {"role": "assistant", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        #max_tokens=1,
        #temperature=0,
    )
    print(f"large_text: {large_text}")
    print(f"small_text: {small_text}")
    print(resp.choices[0].message.content)
    print()
    try:
        result = int(resp.choices[0].message.content.strip())
    except ValueError:
        result = None
    return result

from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
def get_text_embedding(input, client):
    embeddings_batch_response = client.embeddings.create(
        model="mistral-embed",
        inputs=input
    )
    return embeddings_batch_response.data[0].embedding
def compute_similarity(large_text, small_text, client):
    """
    Compute cosine similarity between large and small LLM outputs.
    """

    # Encode the outputs
    embeddings_large = get_text_embedding(large_text, client)
    embeddings_small = get_text_embedding(small_text, client)

    # Compute cosine similarity
    similarity_score = euclidean_distances(np.array(embeddings_large).reshape(1, -1),
                                        np.array(embeddings_small).reshape(1, -1))



    return similarity_score


# ────────────────────────────────────────────────
# MAIN LOOP
# ────────────────────────────────────────────────

DF_FOLDER = "/home/tom/Bureau/phd/mistral_training/hackaton_medi/result_baseline/lwc13mai"
MODE = "lwc"  # "rag" or "lwc"
dfs = [
    pd.read_csv(os.path.join(DF_FOLDER, f))
    for f in os.listdir(DF_FOLDER)
    if f.endswith(".csv")
]
list_judge_labels = []
for df in dfs:
    # ---------- numeric risk‑score agreement ----------
    gt_scores = df["risk_score"].astype(int).tolist()
    gt_recommendations = df["justification"].tolist()

    if MODE == "rag":
        #pred_scores = df["rag_Risk_score"].astype(int).tolist()
        pred_texts = df["rag_recommendation_text"].tolist()
    elif MODE == "lwc":
        #pred_scores = df["lwc_Risk_score"].astype(int).tolist()
        pred_texts = df["lwc_recommendation_text"].tolist()
    else:
        raise ValueError("MODE must be 'rag' or 'lwc'")
    profile_id = df["profile_id"].tolist()

    llm_judge_sync(gt_recommendations[0], pred_texts[0])
    large_text= gt_recommendations[0]
    small_text= pred_texts[0]

    # ---------- LLM‑judge textual agreement ----------
    gt_recommendations = df["justification"].tolist()
    judge_labels = []# large model reference
    list_similarity = []

    for index in tqdm(list(range(0, 303)), desc="Processing judge tasks", total=len(gt_recommendations)):
        b, s  = gt_recommendations[index], pred_texts[index]
        import math
        if type(s)==float and math.isnan(s):
            continue
        retry = True
        couter = 0
        while retry:
            try:
                couter += 1
                res = llm_judge_sync(b, s)
                print(f"Judge result: {res}")
                judge_labels.append(res)
                # Compute similarity
                similarity = compute_similarity(b, s, client)
                print(f"Similarity: {similarity}")
                list_similarity.append(similarity)
                retry = False
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(2)
                if couter > 3:
                    print(f"Error: {e}")
                    retry = False

                if "Requests rate limit exceeded" not in str(e):
                    retry = False

    from pathlib import Path
    np.save(Path(DF_FOLDER).parent /"judge_labels.npy", np.array(judge_labels))
    np.save(Path(DF_FOLDER).parent /"list_similarity.npy", np.array(list_similarity))
    np.save(Path(DF_FOLDER).parent /"profile_id", np.array(profile_id))


    ## plot histogram
    judge_labels_without_none = [x for x in judge_labels if x is not None]
    np.mean(judge_labels_without_none)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.hist(judge_labels_without_none, bins=3, edgecolor='black')
    print(np.mean(judge_labels_without_none))
    print(np.mean(list_similarity))
    # save the figure
    plt.savefig(Path(DF_FOLDER) / "judge_labels_histogram.png")
    plt.show()

    list_similarity = [x[0] for x in list_similarity if x is not None]
    plt.figure(figsize=(10, 6))
    plt.hist(list_similarity, bins=10, edgecolor='black')
    plt.savefig(Path(DF_FOLDER) / "list_similarity.png")
    plt.show()



