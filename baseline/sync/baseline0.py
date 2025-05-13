

import sys
sys.path.append('/hackaton_medi')
import pandas as pd
import requests
from mistralai import Mistral
from baseline.utils import ASSISTANT_INSTRUCTION
import os

def extract_medicine_name(user_query, list_medicine_name):
    # Extract the medicine name from the query
    list_name = []
    list_medicine_name_lower = [name.lower() for name in list_medicine_name]
    user_query_lower = user_query.lower()
    for name in list_medicine_name:
        if name.lower() in user_query.lower():
            list_name.append(name)
    if len(list_name) == 0:
        ## try fo find the closest match
        from difflib import get_close_matches
        list_name = get_close_matches(user_query_lower, list_medicine_name_lower, cutoff=0.1)
    return list_name






def large_context_window_baseline(query, client, model, df_fda_text, test_mode= True):

    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "assistant",
                "content": ASSISTANT_INSTRUCTION
            },
            {
                "role": "user",
                "content": query
                           ,
            },
        ]
    )
    #print(chat_response.choices[0].message.content)

    if chat_response.choices[0].message.content != 'I am not sure I understand the name of  drug, can you be more precise?':
        drug_name = chat_response.choices[0].message.content.split()[0]

        description_patient = chat_response.choices[0].message.content.split(drug_name)[1]
    else:
        if test_mode:
            raise ValueError("The model was not able to extract the drug name from the query.")
        else:
            raise NotImplementedError("The model was not able to extract the drug name from the query.")

    list_medicine_name = df_fda_text['name'].tolist()
    medicine_name = extract_medicine_name(drug_name, list_medicine_name)[0]  # Assuming you want the first match
    description = df_fda_text[df_fda_text['name'] == medicine_name]['fda_text'].values[0]


    chat_response = client.chat.complete(
        model= model,
        messages = [
            {
                "role": "assistant",
                "content":
                    f"You are a helpful assistant to a doctor who wants to prescribe {medicine_name} to a patient.\n\n"
                    f"The doctor describes the patient as follows: {description_patient}\n\n"
                    f"Here is the drug information sheet for {medicine_name}: {description}\n\n"
                    f"First, summarise the risk information sheet.\n\n"
                    f'then, based on the patient description and the drug information, structure your response as follows:\n\n'
                    f"### Recommendation\n"
                    f"Give your professional opinion about prescribing this drug to the patient.\n\n"
                    f"Finally, to finish assign a risk score and present it exactly as:\n"
                    f"#### Risk score: either 0 or 1 or 2\n\n"
                    f"where 0 = appropriate for the patient, 1 = uncertain or potentially dangerous, and 2 = inappropriate "
                    f"and dangerous for the patient.\n\n"
                    f"Do not add any extra commentary or text outside the required formats above.\n\n"

            },
            {
                "role": "user",
                "content": query,
            },
        ]
    )
    summary = chat_response.choices[0].message.content.split("### Recommendation", 1)[0]
    recommendation_text = chat_response.choices[0].message.content.split("### Recommendation", 1)[-1]
    Risk_score = recommendation_text.split("#### Risk score:")[-1].strip()

    #print(chat_response.choices[0].message.content)
    print(f"risk score lwc: {Risk_score}")

    return summary, recommendation_text, Risk_score




if __name__ == "__main__":

    # Load the CSV file
    df_fda_text = pd.read_csv('/hackaton_medi/fda_data/result/drugs_complete_rag.csv')  # Ensure the CSV file is in the same directory or provide the full path


    list_medicine_name = df_fda_text['name'].tolist()

    # Example usage
    user_query = "ACYCLOVIR"
    medicine_name = extract_medicine_name(user_query, list_medicine_name)[0]  # Assuming you want the first match
    description = df_fda_text[df_fda_text['name'] == medicine_name]['fda_text'].values[0]


    api_key = os.getenv("MISTRAL_API_KEY")
    model = "mistral-small-latest"

    client = Mistral(api_key=api_key)

    query = "Should my patient takes 'Acyclovir' knowing he as severe kidney failure"
    test_mode = True

    summary, recommendation_text, Risk_score = large_context_window_baseline(query, client, model, df_fda_text, test_mode= True)

    query = "Should my patient takes the REGADENOSON medicine knowing she as severe caridac failure anti-arrythmic treatment"
    test_mode = True
    summary, recommendation_text, Risk_score = large_context_window_baseline(query, client, model, df_fda_text, test_mode= True)


    #api_key = 'aijcPrPm5pgH78aefnjbxfqy9SOfA7Rj'
    model = "mistral-small-latest"

    client = Mistral(api_key=api_key)


    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "assistant",
                "content": ASSISTANT_INSTRUCTION
            },
            {
                "role": "user",
                "content": query
                ,
            },
        ]
    )
    #print(chat_response.choices[0].message.content)