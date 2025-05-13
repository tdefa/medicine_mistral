

import pandas as pd
import os
import json
import random
import numpy as np
from pathlib import Path





#file_with_csv = "/home/tom/Bureau/phd/mistral_training/hackaton_medi/generation_mock_dataset/genearated_dataset/"
file_with_csv = "/home/tom/Bureau/phd/mistral_training/hackaton_medi/generation_mock_dataset/test_datasets_v2"

# concatenate all csv files in the directory

def concatenate_csv_files(directory):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(os.path.join(directory, file))
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)



if __name__ == "__main__":
    # Concatenate all CSV files in the directory
    concatenated_df = concatenate_csv_files(file_with_csv)
    concatenated_df['id'] = list(concatenated_df.index)
    # Save the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(os.path.join(file_with_csv, 'concatenated_dataset.csv'), index=False)

    print(f"number of unique drugs: {len(concatenated_df['drug_name'].unique())}")
    print(f"number contex: {len(concatenated_df)}")