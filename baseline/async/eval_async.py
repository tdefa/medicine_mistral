
import sys
sys.path.append('/hackaton_medi')
import pandas as pd
import os
import json
import random
import numpy as np
from pathlib import Path

from baseline.baseline0 import large_context_window_baseline
from mistralai import Mistral

import asyncio
from baseline.async_baseline0 import run_many_lcw
import time



if __name__ == "__main__":

    mode = "rag"

    api_key = os.getenv("MISTRAL_API_KEY")

    #model = "open-mistral-7b"
    #model = "ft:open-mistral-7b:86b3ddc9:20250511:aa021324"
    model = "mistral-large-latest"
    model_rag = "decomposition"
    k_rag = 4

    for model in ["mistral-small-latest"]:
        random.seed(0)
        np.random.seed(0)

        if mode=="rag":
            from baseline.async_rag_baseline import run_many_queries
        # Concatenate all CSV files in the directory
        concatenated_df = pd.read_csv(
            "/hackaton_medi/generation_mock_dataset/test_datasets_v2/concatenated_dataset.csv")
        # Save the concatenated DataFrame to a new CSV file
        print(f"number of unique drugs: {len(concatenated_df['drug_name'].unique())}")
        print(f"number contex: {len(concatenated_df)}")
        list_drug_name = concatenated_df['drug_name'].unique()
        columns_name = ['drug_name', 'profile', 'justification', 'summary', 'risk_score']
        try:
            df_test_profile = pd.read_csv("/hackaton_medi/generation_mock_dataset/test_datasets_v2/test_profile0.csv")
        # for each drug name, get one profile for each risk score
        except FileNotFoundError:
            list_test_profile = []
            for drug_name in list_drug_name:

                df_drug = concatenated_df[concatenated_df['drug_name'] == drug_name]
                df_drug = df_drug.sample(frac=1).reset_index(drop=True)

                df_drug = df_drug.drop_duplicates(subset=['risk_score'], keep='first')
                list_test_profile.append(df_drug)

            df_test_profile = pd.concat(list_test_profile, ignore_index=True)
            df_test_profile.to_csv("/home/tom/Bureau/phd/mistral_training/hackaton_medi/"
                                   "generation_mock_dataset/test_datasets_v2/test_profile0.csv", index=False)


        #model = "ministral-3b-latest"
        #model = "mistral-small-latest"
        client = Mistral(api_key=api_key)

        # rag version parameters #########################
        path_databased= "/hackaton_medi/rag_vector/IndexFlatL2"
        drugs_df = pd.read_csv \
            ('/hackaton_medi/fda_data/result/drugs_complete_rag.csv')

        # large context window parameters #########################
        df_fda_text = pd.read_csv \
            ('/hackaton_medi/fda_data/result/drugs_complete.csv')


        # Load the CSV file
        list_exeption_rag = []
        list_exeption_lcw = []
        list_dict_res = []

        # iterate over the rows of the DataFrame df_test_profile
        import tqdm
        loop = tqdm.tqdm(total=len(df_test_profile), desc="Processing", unit="drug")
        list_input_async = []

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

            list_input_async.append(
                {
                    "drug_name": drug_name,
                    "profile": profile,
                    "gr_justification": gr_justification,
                    "gr_summary": gr_summary,
                    "gr_risk_score": gr_risk_score,
                    "profile_id": profile_id,
                    "query": query,
                    "description": description
                }
            )

        if mode == "rag":
            t = time.time()  # start timer
            all_results = asyncio.run(
                run_many_queries(
                    list_input_async=list_input_async,
                    model=model,
                    drugs_df=drugs_df,
                    api_key = 'dY1K9ZuD1FF4ELnyEkCiZKuuGpP2ZQBK',
                    max_concurrency=3,   # ≤ 5 concurrent calls at any moment
                    k_rag=k_rag,
                    model_rag=model_rag
                )
            )
            print("Elapsed time:", time.time() - t)
        elif mode == "lwc":
            t = time.time()
            all_results = asyncio.run(
                run_many_lcw(
                    list_input_async=list_input_async,
                    model=model,
                    df_fda_text=df_fda_text,
                    api_key=api_key,
                    max_concurrency=2,  # ≤ 5 concurrent calls at any moment
                )
            )
        else:
            raise ValueError("Invalid mode. Choose either 'rag' or 'lcw'.")

        import datetime
        import math
        now = datetime.datetime.now()
        str_now = now.strftime("%Y-%m-%d %H:%M:%S")
        curated_list_results = []
        for res in all_results:
            if res is None:
                continue
            if None in res:
                continue
            if type(res["summary"])==float and math.isnan(res["summary"]):
                res["summary"] = ""
            curated_list_results.append(res)
        df_res = pd.DataFrame(curated_list_results)
        if mode=="rag":
            df_res.to_csv(f"/home/tom/Bureau/phd/mistral_training/hackaton_medi/result_baseline/res_{mode}_{model_rag}_{model}krag{k_rag}_{str_now}.csv", index=False)
        elif mode=="lwc":
            df_res.to_csv(f"/home/tom/Bureau/phd/mistral_training/hackaton_medi/result_baseline/res_{mode}_{model}_{str_now}.csv", index=False)

        list_tokens = []
        for index, row in df_res.iterrows():
            tok = row['tokens']
            list_tokens.append(tok)

    # remove nan
        list_tokens = [x for x in list_tokens if str(x) != 'nan']
        print(np.sum(list_tokens)/len(list_tokens))