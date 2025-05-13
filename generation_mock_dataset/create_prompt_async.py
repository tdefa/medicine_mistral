"""
async_profiles.py  •  2025‑05‑11

Asynchronous rewrite of the original patient‑profile generator.
—
• aiofiles      → async file I/O
• httpx         → async HTTP client (used for Mistral’s REST API)
• asyncio       → concurrency with gather() + Semaphore
• tqdm.asyncio  → non‑blocking progress bar
"""

import os
import re
import random
import asyncio
from pathlib import Path
from typing import List, Tuple, Dict, Any

import aiofiles
import httpx
import pandas as pd
from tqdm.asyncio import tqdm  # note: tqdm>=4.66 has asyncio support
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from tqdm.asyncio import tqdm_asyncio        # progress bar for async loops

# CONFIG
API_KEY         = os.environ.get("MISTRAL_API_KEY")
MODEL_NAME      = "mistral-large-latest"
CONCURRENCY     = 6                     # how many simultaneous API calls
MAX_DRUGS       = 100                    # upper‑bound of union_list to process
SAVE_DIR        = Path("/home/tom/Bureau/phd/mistral_training/hackaton_medi")
OUTPUT_SUBDIR   = SAVE_DIR / "generation_mock_dataset/test_datasets_v2"

# Pre‑load static assets synchronously (fast & simple) --------------------------
profiles_path   = SAVE_DIR / "generation_mock_dataset/profile_example.txt"
unitox_df       = pd.read_csv(SAVE_DIR / "14042913/UniTox.csv")
drugs_df        = pd.read_csv(SAVE_DIR / "fda_data/result/drugs_complete.csv")


tokenizer       = MistralTokenizer.from_model(MODEL_NAME)

list_name_unitox = unitox_df['Generic Name'].tolist()
list_name_fda    = drugs_df['name'].tolist()
union_list = list(set(list_name_unitox).intersection(set(list_name_fda)))
random.shuffle(union_list)

# already‑generated drugs (avoid duplicates or data leakage)
df_done_drugs = pd.read_csv("/home/tom/Bureau/phd/mistral_training/hackaton_medi/dataset_finetune/concatenated_dataset.csv")

DONE_DRUGS = set(df_done_drugs['drug_name'].unique())

# Low‑level helpers

async def extract_patient_profiles(path: Path) -> str:
    """Asynchronously read example profiles from txt file."""
    async with aiofiles.open(path, mode='r') as fh:
        content = await fh.read()
    # double‑newline split then re‑stitch with single newlines (original logic)
    profiles = content.strip().split('\n\n')
    return '\n '.join(profiles)


def get_summary(drugs: pd.DataFrame, drug_name: str) -> str:
    return drugs.iloc[list_name_fda.index(drug_name)]['fda_text']


def get_toxicity(unitox: pd.DataFrame, drug_name: str
                 ) -> Tuple[Dict[str, int], Dict[str, str]]:
    tox_keys = [
        "Cardiotoxicity", "DermatologicalToxicity", "HematologicalToxicity",
        "LiverToxicity", "PulmonaryToxicity", "RenalToxicity", "Infertility"
    ]
    row = unitox.iloc[union_list.index(drug_name)]

    rate, summary = {}, {}
    for key in tox_keys:
        rating = row[f"{key} Ternary Rating"]
        rate[key] = {"Most": 2, "Less": 1, "No": 0}.get(rating, 0)
        summary[key] = row[f"{key} Reasoning"]

    return rate, summary


def sample_toxicity_rate(rate: Dict[str, int]
                         ) -> Tuple[List[str], List[str], List[str]]:
    """Return (safe, medium, dangerous) lists of toxicity names."""
    lists = {0: [], 1: [], 2: []}
    for key, val in rate.items():
        lists[val].append(key.split(" ")[0])

    for lst in lists.values():
        random.shuffle(lst)
    return lists[0], lists[1], lists[2]


def get_profile_lines(patient_profiles: str, drug_name: str,
                      summary: str, risk_score: int
                      ) -> Tuple[List[Dict[str, Any]], bool]:
    """Validate / parse AI output into profile dicts."""
    matches = re.findall(r'\*\*(.*?)\*\*', patient_profiles)
    parsed, global_valid = [], True

    for i in range(0, len(matches), 2):
        try:
            profile, justification = matches[i], matches[i + 1]
        except IndexError:
            break

        if ("Patient factual information" not in profile or len(profile) < 30 or
                "Reason why the drug is indicated" not in justification or len(justification) < 53):
            global_valid = False
            continue

        parsed.append({
            "drug_name":   drug_name,
            "profile":     profile[2:],   # strip '**'
            "justification": justification,
            "summary":     summary,
            "risk_score":  risk_score
        })

    return parsed, bool(parsed and global_valid)




# ────────────────────────────────────────────────────────────────────────────────
# Core async profile generator
# ────────────────────────────────────────────────────────────────────────────────
async def generate_profiles(
        client,
        drug_name: str,
        example_profiles: str,
) -> Tuple[List[Dict[str, Any]], bool, str, List[bool]]:

    print('fetching drug info for', drug_name)

    description                         = get_summary(drugs_df, drug_name)
    rate, dict_toxicity_summary                   = get_toxicity(unitox_df, drug_name)
    tox0, tox1, tox2                    = sample_toxicity_rate(rate)

    if not tox2 or not tox0:
        return None, False, f"Drug {drug_name} lacks required toxicity distribution.", None

    selected_toxicity = [tox0.pop(0)]                        # at least one “safe”
    while len(selected_toxicity) < 4 and (tox2 or tox1 or tox0):
        for source in (tox2, tox1, tox0):
            if source and len(selected_toxicity) < 4:
                selected_toxicity.append(source.pop(0))

    # Build prompt ----------------------------------------------------------------

    promt = (f"You are a clinician preparing short medical case profiles for your students.\n"
             "Here are some examples of patient profiles:\n"
             f"{example_profiles}"
             "Your task is to write new patient profiles that follow these rules:\n"
             "- Each profile should clearly describe the patient’s health condition in fewer than 150 words.\n"
             f"- Based on the patient's condition, the drug {drug_name} should be categorized as either:\n"
             "(1) indicated and appropriate,\n"
             "(2) potentially dangerous\n"
             "(3) dangerous, contraindicated and inappropriate.\n"
             "Below is the official description (notice) of the drug {drug_name}:\n"
             f"{description}\n"
             "You may also use the following additional information to help you:\n")

    for toxicity in selected_toxicity:
        promt += f" Here is the summary for the {toxicity}: {dict_toxicity_summary[toxicity]} \n"
        promt += f"The {toxicity} is rated as {rate[toxicity]} /2\n"
    promt += f"\nFirst summarize the risk of the drug {drug_name} and then answer to the user request  \n"

    messages = [
        {
            "role": "system",
            "content": f"{promt}"
        },
        {
            "role": "user",
            "content": f"""give me a 5 short patient profiles (between 20 and 150 world) for each of the following categories: 
                            (1) The {drug_name} is not dangerous, indicated and appropriate,
                            (2) The {drug_name} is potentially dangerous
                            (3) The {drug_name} is dangerous, contraindicated and inappropriate."""
                       f"moreover, start by factual information about the patient, "
                       f" then separate the information with ** and then give the reason why the drug is contraindicated."
                       f"structure your answer in the following way: "
                       f"### Patient Profiles Categorie 1 \n"
                       f"**1.Patient factual information : {{patient profile here}} ** \n"
                       f"**Reason why the drug is indicated or contraindicated: {{Reason why here}} ** \n"
                       f"**1.Patient factual information : {{patient profile here}} ** \n"
                       f"**Reason why the drug is indicated or contraindicated: {{Reason why here}} ** \n"
                       f" ect ..."
                       f"respect the format so it can be easy to parse and do not add any other information."
        }
    ]

    # Token length guard — stays synchronous (fast)
    tokens = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages))
    if len(tokens.tokens) > 70_000:
        return None, False, "Prompt too long.", None

    # API call ---------------------------------------------------------------------
    chat_response = await  client.chat.complete_async(
        model=model,
        messages=messages)
    text = chat_response.choices[0].message.content


# Parse / validate -------------------------------------------------------------
    summary = text.split("### Patient Profiles Category")[0]
    valid   = "summary" in summary.lower()
    list_valid_profile_cat = [False, False, False]
    profiles: List[Dict[str, Any]] = []

    try:
        cat1 = text.split("### Patient Profiles Category 1")[1].split("### Patient Profiles Category 2")[0]
        prof_cat1, valid1 = get_profile_lines(cat1, drug_name, summary, 0)
        if valid1: profiles += prof_cat1
        list_valid_profile_cat[0] = valid1
    except Exception as e:
        print(f"Error parsing category 1: {e}")
        list_valid_profile_cat[0] = False

    try:
        cat2 = text.split("### Patient Profiles Category 2")[1].split("### Patient Profiles Category 3")[0]
        prof_cat2, valid2 = get_profile_lines(cat2, drug_name, summary, 1)
        if valid2: profiles += prof_cat2
        list_valid_profile_cat[1] = valid2
    except Exception as e:
        print(f"Error parsing category 2: {e}")
        list_valid_profile_cat[1] = False

    try:
        cat3 = text.split("### Patient Profiles Category 3")[1]
        prof_cat3, valid3 = get_profile_lines(cat3, drug_name, summary, 2)
        if valid3: profiles += prof_cat3
        list_valid_profile_cat[2] = valid3
    except Exception as e:
        print(f"Error parsing category 3: {e}")
        list_valid_profile_cat[2] = False

    valid = valid and all(list_valid_profile_cat) and bool(profiles)
    return profiles or None, valid, text, list_valid_profile_cat


# ────────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────────────
async def process_drug(drug_name: str,
                       semaphore: asyncio.Semaphore,
                       example_profiles: str,
                       client) -> List[Dict[str, Any]]:
    """Semaphore‑guarded wrapper so we never exceed CONCURRENCY."""
    if drug_name in DONE_DRUGS:
        return []

    async with semaphore:
        profiles, valid, text, list_valid_profile_cat = await generate_profiles(client, drug_name, example_profiles)
    return profiles or []


async def main(client) -> None:
    example_profiles = await extract_patient_profiles(profiles_path)
    sem             = asyncio.Semaphore(CONCURRENCY)

    tasks = [
        process_drug(drug, sem, example_profiles, client)
        for drug in union_list[:MAX_DRUGS]
        if drug not in DONE_DRUGS
    ]

    results = await tqdm_asyncio.gather(*tasks)

    # Flatten & persist -----------------------------------------------------------
    final_profiles = [item for sub in results for item in sub]
    if not final_profiles:
        print("No valid profiles generated.")
        return

    df = pd.DataFrame(final_profiles)
    OUTPUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    rand_id = random.randint(0, 100_000)
    outfile = OUTPUT_SUBDIR / f"profile_{rand_id}.csv"
    df.to_csv(outfile, index=False)
    print(f"Saved {len(df)} profiles → {outfile}")


if __name__ == "__main__":

    from mistralai import Mistral
    import pandas as pd
    import random
    from tqdm import tqdm
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    api_key = 'dY1K9ZuD1FF4ELnyEkCiZKuuGpP2ZQBK'
    model = "mistral-large-latest"
    client = Mistral(api_key=api_key)

    asyncio.run(main(client))


    #df = pd.read_csv("/home/tom/Bureau/phd/mistral_training/hackaton_medi/generation_mock_dataset/genearated_dataset/concatenated_dataset.csv")
    df = pd.read_csv(f"/home/tom/Bureau/phd/mistral_training/hackaton_medi/generation_mock_dataset/test_datasets_v2/profile_59633.csv")

    ## iterate over the rows of the dataframe
    list_input_async = []
    for index, row in df.iterrows():
        # Extract the values of the columns
        drug_name = row['drug_name']
        profile = row['profile']
        justification = row['justification']
        summary = row['summary']
        risk_score = row['risk_score']
        #profile_id = row['id']

        # Create the query
        query = f"Should my patient takes following drug '{drug_name.title()}', knowing its profile : {profile}?"

        # Get the description of the drug from the drugs_df DataFrame
        description = drugs_df[drugs_df['name'] == drug_name]['fda_text'].values[0]

        list_input_async.append(
            {
                "drug_name": drug_name,
                "profile": profile,
                "justification": justification,
                "summary": summary,
                "risk_score": risk_score,
                "query": query,
                "description": description
            }
        )
