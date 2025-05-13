
from mistralai import Mistral
import pandas as pd
import random
from tqdm import tqdm
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import numpy as np
import json
medicine_name  = "test"

if __name__ == '__main__':

    drugs_df = pd.read_csv \
        ('/home/tom/Bureau/phd/mistral_training/hackaton_medi/fda_data/result/drugs_complete.csv')
    list_name_fda = drugs_df['name'].tolist()
    concatenated_df = pd.read_csv(
        "/home/tom/Bureau/phd/mistral_training/hackaton_medi/dataset_finetune/concatenated_dataset.csv")
    # Save the concatenated DataFrame to a new CSV file
    print(f"number of unique drugs: {len(concatenated_df['drug_name'].unique())}")
    print(f"number contex: {len(concatenated_df)}")
    list_drug_name = concatenated_df['drug_name'].unique()
    columns_name = ['drug_name', 'profile', 'justification', 'summary', 'risk_score']
    ## for each drug name, get one profile for each risk score
    try:
        df_test_profile = pd.read_csv(
            "/home/tom/Bureau/phd/mistral_training/hackaton_medi/dataset_finetune/concatenated_dataset.csv")
    except FileNotFoundError:
        list_test_profile = []
        for drug_name in list_drug_name:

            df_drug = concatenated_df[concatenated_df['drug_name'] == drug_name]
            df_drug = df_drug.sample(frac=1).reset_index(drop=True)

            df_drug = df_drug.drop_duplicates(subset=['risk_score'], keep='first')
            list_test_profile.append(df_drug)

        df_test_profile = pd.concat(list_test_profile, ignore_index=True)
        df_test_profile.to_csv \
            ("/home/tom/Bureau/phd/mistral_training/hackaton_medi/dataset_finetune/concatenated_dataset.csv", index=False)



    test_dataset = pd.read_csv("/home/tom/Bureau/phd/mistral_training/hackaton_medi/generation_mock_dataset/test_datasets_v2/concatenated_dataset.csv")
    list_drug = test_dataset['drug_name'].unique()


    import math
    ## generate jsons form df
    list_jsonl = []
    dict_json = {}
    nb_share_drug = 0
    for index, row in df_test_profile.iterrows():
        drug_name = row['drug_name']
        if drug_name in list_drug:
            nb_share_drug += 1
        description_patient = row['profile']
        gr_justification = row['justification']
        gr_summary = row['summary']
        gr_risk_score = row['risk_score']
        #profile_id = row['id']
        # break
        description = drugs_df.iloc[list_name_fda.index(drug_name)]['fda_text']
        query = f"Should my patient take  {drug_name} ? Knowing this short patient profile: {description_patient}"

        if type(description)==float and math.isnan(description):
            print(f"error with {drug_name}")
            continue


        message = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"You are a helpful assistant to a doctor who wants to prescribe {drug_name} to a patient.\n\n"
                        f"The doctor describes the patient as follows: {description_patient}\n\n"
                        f"Here is the drug information sheet for {drug_name}:\n {description}\n\n"
                        f"First, summarise the risk information sheet.\n\n"
                        f"then, based on the patient description and the drug information, structure your response as follows:\n\n"
                        f"### Recommendation\n"
                        f"Give your professional opinion about prescribing this drug to the patient.\n\n"
                        f"Finally, to finish assign a risk score and present it exactly as:\n"
                        f"#### Risk score: either 0 or 1 or 2\n\n"
                        f"where 0 = appropriate for the patient, 1 = uncertain or potentially dangerous, and 2 = inappropriate "
                        f"and dangerous for the patient.\n\n"
                        f"Do not add any extra commentary or text outside the required formats above.\n\n"
                    ),
                },
                {
                    "role": "user",
                    "content": query
                },
                {
                    "role": "assistant",
                    "content": (
                        f"{gr_summary}\n\n"
                        f"### Recommendation \n"
                        f"{gr_justification}\n\n"
                        f"#### Risk score: {gr_risk_score}\n"
                    ),
                }
            ]
        }

        # Convert the dictionary to a JSON string
        json_message = json.dumps(message)

        list_jsonl.append(json_message)

    import json

    with open('/home/tom/Bureau/phd/mistral_training/hackaton_medi/dataset_finetune/train_profile.jsonl', 'w') as f:
        for item in list_jsonl[:267]:
            f.write(json_message+"\n")

    ## save jsonl
    with open('/home/tom/Bureau/phd/mistral_training/hackaton_medi/dataset_finetune/eval_profile.jsonl', 'w') as f:
        for item in list_jsonl[267:]:
            f.write(json_message+"\n")




    ## upload datasets

    from mistralai import Mistral
    import os

    api_key = os.environ.get("MISTRAL_API_KEY")   # put yours here or in env
    client = Mistral(api_key=api_key)



    ultrachat_chunk_train = client.files.upload(file={
        "file_name": "train_profile.jsonl",
        "content": open("/home/tom/Bureau/phd/mistral_training/hackaton_medi/dataset_finetune/train_profile.jsonl", "rb"),
    })
    ultrachat_chunk_eval = client.files.upload(file={
        "file_name": "eval_profile.jsonl",
        "content": open('/home/tom/Bureau/phd/mistral_training/hackaton_medi/dataset_finetune/eval_profile.jsonl', "rb"),
    })
