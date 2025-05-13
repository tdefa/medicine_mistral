import os
from mistralai import Mistral
import pandas as pd
import random
from tqdm import tqdm
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest


# Read the file and extract patient profiles
def extract_patient_profiles(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    # Split the content into individual profiles based on double newlines
    profiles = content.strip().split('\n\n')
    str_profiles = '\n '.join(profiles)
    return str_profiles


def get_summary(drugs_df, drug_name):
    # Extract the medicine name from the query

    description = drugs_df.iloc[list_name_fda.index(drug_name)]['fda_text']

    return description


def get_toxicity(unitox_df, drug_name):
    # Extract the medicine name from the query
    list_toxicity = ["Cardiotoxicity",
                     "DermatologicalToxicity",
                     "HematologicalToxicity",
                     "LiverToxicity",
                     "PulmonaryToxicity",
                     "RenalToxicity",
                     "Infertility", ]

    dict_toxicity_rate = {}
    dict_toxicity_summary = {}

    unitox_df.iloc[union_list.index(drug_name)]

    for key in list_toxicity:
        rating = unitox_df.iloc[union_list.index(drug_name)][key + " Ternary Rating"]
        if rating == "Most":
            rating = 2
        elif rating == "Less":
            rating = 1
        elif rating == "No":
            rating = 0
        dict_toxicity_rate[key] = rating

        dict_toxicity_summary[key] = unitox_df.iloc[union_list.index(drug_name)][key + " Reasoning"]

    return dict_toxicity_rate, dict_toxicity_summary


# give the list of most important toxicity
def sample_toxicity_rate(dict_toxicity_rate):
    toxicity_list_2 = []
    toxicity_list_1 = []
    toxicity_list_0 = []
    for key, value in dict_toxicity_rate.items():
        if value == 2:
            toxicity_list_2.append(key.split(" ")[0])
        elif value == 1:
            toxicity_list_1.append(key.split(" ")[0])
        elif value == 0:
            toxicity_list_0.append(key.split(" ")[0])
        else:
            raise ValueError(f"Unexpected value {value} for key {key}")
        random.shuffle(toxicity_list_2)
        random.shuffle(toxicity_list_1)
        random.shuffle(toxicity_list_0)
    return toxicity_list_0, toxicity_list_1, toxicity_list_2


if __name__ == "__main__":



    def get_profile_lines(patient_profiles, drug_name, summary, risk_score):
        # Extract the lines from the patient profiles
        # Filter out empty lines
        import re
        matches = re.findall(r'\*\*(.*?)\*\*', patient_profiles)
        list_profile = []
        valid_format = True
        # Group into profiles and contraindications
        for i in range(0, len(matches), 2):
            profile = matches[i][2:]

            if not "Patient factual information" in profile or len(profile) < 30:
                valid_format = False
                continue
            justification = matches[i + 1]
            if not "Reason why the drug is indicated or contraindicated" in justification or len(justification) < 53:
                valid_format = False
                continue
            if valid_format:
                line_dict = {"drug_name": drug_name, "profile": profile,
                             "justification": justification, "summary": summary,
                             "risk_score": risk_score}
                list_profile.append(line_dict)
        return list_profile, valid_format


    def generate_profiles(client, drug_name, tokenizer):
        # Generate a profile for a drug with a score of 0,1,2
        description = get_summary(drugs_df, drug_name)
        dict_toxicity_rate, dict_toxicity_summary = get_toxicity(unitox_df, drug_name)

        toxicity_list_0, toxicity_list_1, toxicity_list_2 = sample_toxicity_rate(dict_toxicity_rate)
        if len(toxicity_list_2) > 1:
            print(f"Drug {drug_name} has no toxicity. Skipping...")
            return None, None, f"Drug {drug_name} has no toxicity. Skipping...", None
        selected_toxicity = []
        while len(selected_toxicity) < 3:
            if len(toxicity_list_2) > 0:
                selected_toxicity.append(toxicity_list_2.pop(0))
            if len(toxicity_list_1) > 0:
                selected_toxicity.append(toxicity_list_1.pop(0))
            if len(toxicity_list_0) > 0:
                selected_toxicity.append(toxicity_list_0.pop(0))

        if len(toxicity_list_0) > 1:
            selected_toxicity.append(toxicity_list_0.pop(0))

        else:
            print(f"Drug {drug_name} has is full toxicity. Skipping...")
            return None, None, f"Drug {drug_name} has is full toxicity. Skipping...", None


        if len(toxicity_list_1) > 0:
            selected_toxicity.append(toxicity_list_1.pop(0))

        promt = f"""You are a clinician preparing short medical case profiles for your students.
                            
                            Here are some examples of patient profiles:
                            {profiles}
                            
                            Your task is to write new patient profiles that follow these rules:
                            - Each profile should clearly describe the patient’s health condition in fewer than 150 words.
                            - Based on the patient's condition, the drug {drug_name} should be categorized as either:
                            (1) indicated and appropriate,
                            (2) potentially dangerous
                            (3) dangerous, contraindicated and inappropriate.
                            
                            Below is the official description (notice) of the drug {drug_name}:
                            {description}
                    
                    You may also use the following additional information to help you:
                    """

        for toxicity in selected_toxicity:
            promt += f" Here is the summary for the {toxicity}: {dict_toxicity_summary[toxicity]} \n"
        promt += f"first summarize the risk of the drug {drug_name} and then answer to the user request  \n"

        messages = [
            {
                "role": "assistant",
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
        tokens = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages))
        if len(tokens.tokens) > 50000:
            print("The prompt is too long. Please shorten it.")
            return None, None, "The prompt is too long. Please shorten it.", None

        chat_response = client.chat.complete(
            model=model,
            messages=messages)
        #
        print(chat_response.choices[0].message.content)

        text = chat_response.choices[0].message.content
        valid_format = True
        summary = text.split("### Patient Profiles Category")[0]

        if not "summary" in summary.lower():
            print("The summary is not valid. Please check the output.")
            valid_format = False
            return None, valid_format, text, None


    # Extract the lines from the patient profiles
        list_profile = []
        try:
            patient_profiles_cat0 = \
            text.split("### Patient Profiles Category 1")[1].split("### Patient Profiles Category 2")[0]
            list_profile_cat0, valid_format_cat0 = get_profile_lines(patient_profiles_cat0, drug_name, summary,
                                                                     risk_score=0)
            if valid_format_cat0:
                list_profile += list_profile_cat0
        except Exception as e:
            print("Error in extracting category 0 profiles:", e)
            valid_format = False
            valid_format_cat0 = False

        try:
            patient_profiles_cat1 = \
                text.split("### Patient Profiles Category 2")[1].split("### Patient Profiles Category 3")[0]
            list_profile_cat1, valid_format_cat1 = get_profile_lines(patient_profiles_cat1, drug_name, summary,
                                                                     risk_score=1)
            if valid_format_cat1:
                list_profile += list_profile_cat1
        except Exception as e:
            print("Error in extracting category 1 profiles:", e)
            valid_format = False
            valid_format_cat1 = False
        try:
            patient_profiles_cat2 = \
            text.split("### Patient Profiles Category 3")[1]
            list_profile_cat2, valid_format_cat2 = get_profile_lines(patient_profiles_cat2, drug_name, summary,
                                                                      risk_score=2)
            if valid_format_cat2:
                list_profile += list_profile_cat2
        except Exception as e:
            print("Error in extracting category 2 profiles:", e)
            valid_format = False
            valid_format_cat2 = False

        # Check if the format is valid
        if not valid_format_cat0 or not valid_format_cat1 or not valid_format_cat2:
            print("The format is not valid. Please check the output.")
            valid_format = False
        list_valid_profile_cat = [valid_format_cat0, valid_format_cat1, valid_format_cat2]
        return list_profile, valid_format, text, list_valid_profile_cat


    # drug_name = "ALPELISIB"
    # Extract the medicine name from the query
    # drugs_df.iloc[union_list.index(drug_name)]
    # unitox_df.iloc[union_list.index(drug_name)]

    # GENEARATE PROMPT FOR SCROE 2

    model_name = "mistral-large-latest"

    api_key = os.environ.get("MISTRAL_API_KEY")
    model = "mistral-large-latest"
    client = Mistral(api_key=api_key)
    tokenizer = MistralTokenizer.from_model(model_name)



    # Read the CSV file into a DataFrame
    file_path = '/home/tom/Bureau/phd/mistral_training/hackaton_medi/generation_mock_dataset/profile_example.txt'
    # Example usage
    profiles = extract_patient_profiles(file_path)

    unitox_df = pd.read_csv('/home/tom/Bureau/phd/mistral_training/hackaton_medi/14042913/UniTox.csv')
    list_name_unitox = unitox_df['Generic Name'].tolist()
    print(unitox_df.columns)

    drugs_df = pd.read_csv \
        ('/home/tom/Bureau/phd/mistral_training/hackaton_medi/fda_data/result/drugs_complete.csv')
    # Ensure the CSV file is in the same directory or provide the full path
    list_name_fda = drugs_df['name'].tolist()
    union_list = list(set(list_name_unitox).intersection(set(list_name_fda)))
    random.shuffle(union_list)
    list_valide_format = []
    list_unvalid_format = []
    list_valide_text = []
    list_unvalid_text = []
    list_unvalid_profile_cat = []
    list_invalid_text = []
    concatenated_df = pd.read_csv("/home/tom/Bureau/phd/mistral_training/hackaton_medi/generation_mock_dataset/genearated_dataset/concatenated_dataset.csv")
    # Save the concatenated DataFrame to a new CSV file
    print(f"number of unique drugs: {len(concatenated_df['drug_name'].unique())}")
    print(f"number contex: {len(concatenated_df)}")
    list_drug_name = concatenated_df['drug_name'].unique()
    for i in tqdm(list(range(0, 100))):
        drug_name = union_list[i]
        print(drug_name)
        if drug_name in list_drug_name:
            continue
        # Generate a profile for a drug with a score of 2
        profile, valid_format, text, list_valid_profile_cat = generate_profiles(client, drug_name, tokenizer)
        if profile is None:
            print("The profile is None. Please check the output.")
            valid_format = False
        if valid_format:
            list_valide_format.append(profile)
        else:
            list_unvalid_format.append(profile)
            list_unvalid_profile_cat.append(list_valid_profile_cat)
            list_invalid_text.append(text)
        if len(list_valide_format) == 3:
            break
    final_list = []
    for l in list_valide_format[:]:
        final_list += l

    df = pd.DataFrame(final_list)
    random_int = random.randint(0, 100000)
    df.to_csv(f'/home/tom/Bureau/phd/mistral_training/hackaton_medi/'
              f'generation_mock_dataset/genearated_dataset/profile_{random_int}.csv', index=False)
