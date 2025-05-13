
import sys
sys.path.append('/hackaton_medi')
import pandas as pd
import os
import json
import random
import numpy as np
from pathlib import Path

from baseline.rag_baseline import rag_baseline
from baseline.baseline0 import large_context_window_baseline
from mistralai import Mistral





if __name__ == "__main__":
    mode = "rag"
    # Concatenate all CSV files in the directory
    concatenated_df = pd.read_csv("/hackaton_medi/generation_mock_dataset/genearated_dataset/concatenated_dataset.csv")
    # Save the concatenated DataFrame to a new CSV file
    print(f"number of unique drugs: {len(concatenated_df['drug_name'].unique())}")
    print(f"number contex: {len(concatenated_df)}")
    list_drug_name = concatenated_df['drug_name'].unique()
    columns_name = ['drug_name', 'profile', 'justification', 'summary', 'risk_score']
    ## for each drug name, get one profile for each risk score
    list_test_profile = []
    for drug_name in list_drug_name:

        df_drug = concatenated_df[concatenated_df['drug_name'] == drug_name]
        df_drug = df_drug.sample(frac=1).reset_index(drop=True)

        df_drug = df_drug.drop_duplicates(subset=['risk_score'], keep='first')
        list_test_profile.append(df_drug)

    df_test_profile = pd.concat(list_test_profile, ignore_index=True)
    df_test_profile.to_csv("/home/tom/Bureau/phd/mistral_training/hackaton_medi/generation_mock_dataset/genearated_dataset/test_profile0.csv", index=False)

    api_key = os.getenv("MISTRAL_API_KEY")

    model = "mistral-small-latest"
    #model = "ministral-3b-latest"
    client = Mistral(api_key=api_key)

    # rag version parameters #########################
    path_databased= "/hackaton_medi/rag_vector/IndexFlatL2"
    drugs_df = pd.read_csv \
        ('/hackaton_medi/fda_data/result/drugs_complete_rag.csv')
    model_rag = "mistral-small-latest"

    # large context window parameters #########################
    df_fda_text = pd.read_csv \
        ('/hackaton_medi/fda_data/result/drugs_complete.csv')
    model_lcw = "mistral-small-latest"


    # Load the CSV file
    list_exeption_rag = []
    list_exeption_lcw = []
    list_dict_res = []

    # iterate over the rows of the DataFrame df_test_profile
    import tqdm
    loop = tqdm.tqdm(total=len(df_test_profile), desc="Processing", unit="drug")

    for index, row in df_test_profile.iterrows():
        loop.update(1)
        # Extract the values of the columns
        drug_name = row['drug_name']
        profile = row['profile']
        gr_justification = row['justification']
        gr_summary = row['summary']
        gr_risk_score = row['risk_score']
        profile_id = row['id']

        profile=profile.split("Patient factual information:")[-1]

        # Create the query
        query = f"Should my patient takes following drug '{drug_name.title()}', knowing its profile : {profile}?"

        # Get the description of the drug from the drugs_df DataFrame
        description = drugs_df[drugs_df['name'] == drug_name]['fda_text'].values[0]

        # Call the rag_baseline function
        if mode=="rag":
            try:
                rag_summary, rag_recommendation_text, rag_Risk_score, retrieved_chunk = rag_baseline(query,
                                                                                    client, model,
                                                                        chunk_size=60,
                                                                        test_mode=True,
                                                                        path_databased=path_databased,
                                                                        drugs_df=drugs_df)
            except Exception as e:
                print(f"Error in RAG baseline for drug {drug_name}: {e}")
                rag_summary, rag_recommendation_text, rag_Risk_score = None, None, None
                list_exeption_rag.append((drug_name, e))
            lwc_summary, lwc_recommendation_text, lwc_Risk_score = None, None, None

        elif mode=="lwc":
            try:
                lwc_summary, lwc_recommendation_text, lwc_Risk_score = large_context_window_baseline(query,
                                                                                                     client,
                                                                                                     model,
                                                                                                     df_fda_text=df_fda_text,
                                                                                                     test_mode=True)

            except Exception as e:
                print(f"Error in LCW baseline for drug {drug_name}: {e}")
                lwc_summary, lwc_recommendation_text, lwc_Risk_score = None, None, None
                list_exeption_lcw.append((drug_name, e))
            rag_summary, rag_recommendation_text, rag_Risk_score, retrieved_chunk = None, None, None, None
        else:
            raise ValueError("Invalid mode. Choose either 'rag' or 'lcw'.")


        dict_res = {
            "drug_name": drug_name,
            "profile": profile,
            "justification": gr_justification,
            "summary": gr_summary,
            "risk_score": gr_risk_score,
            "rag_summary": rag_summary,
            "rag_recommendation_text": rag_recommendation_text,
            "rag_Risk_score": rag_Risk_score,
            "lwc_summary": lwc_summary,
            "lwc_recommendation_text": lwc_recommendation_text,
            "lwc_Risk_score": lwc_Risk_score,
            "profile_id": profile_id,
            "retrieved_chunk": retrieved_chunk
        }
        list_dict_res.append(dict_res)

        print(drug_name, gr_risk_score, rag_Risk_score, lwc_Risk_score)
        #flflf
    import datetime
    now = datetime.datetime.now()
    str_now = now.strftime("%Y-%m-%d %H:%M:%S")
    df_res = pd.DataFrame(list_dict_res)
    if mode=="rag":
        df_res.to_csv(f"/home/tom/Bureau/phd/mistral_training/hackaton_medi/result_baseline/res_{mode}_test_{str_now}.csv", index=False)

    elif mode=="lwc":
        df_res.to_csv(f"/home/tom/Bureau/phd/mistral_training/hackaton_medi/result_baseline/res_{mode}_test_{str_now}.csv", index=False)

    else:
        raise Exception('mode not recognized')



