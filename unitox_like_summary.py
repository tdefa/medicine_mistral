

import pandas as pd
import os
import json

def read_json(path):
    with open(path, 'r') as file:
        return json.load(file)

def get_fda_text(drug_name, spls, fields_fda, rag_mode = False):
    text = ""
    for field in fields_fda:
        txt = spls[drug_name].get(field, "")
        if len(txt) == 1:
            field = field.replace("_", " ")
            test_to_add = f"\n <section> <section name>{field}</section name>" + txt[0] +"</section> \n \n"
            if rag_mode:
                text += test_to_add
            else:
                text += txt[0]
        else:
            print(txt)
    return text

fields_fda = [
    'indications_and_usage',
    'dosage_and_administration',
    'description',
    'adverse_reactions',
    'contraindications',
    'clinical_pharmacology',
    'pregnancy',
    'pediatric_use',
    'overdosage',
    'carcinogenesis_and_mutagenesis_and_impairment_of_fertility',
    'information_for_patients',
    'drug_interactions',
    'geriatric_use',
    'warnings_and_cautions',
    'dosage_forms_and_strengths',
    'use_in_specific_populations',
    'nonclinical_toxicology',
    'clinical_studies',
    'pharmacokinetics',
    'mechanism_of_action',
    'adverse_reactions_table',
    'spl_unclassified_section',
    'pharmacodynamics',
    'warnings',
    'precautions',
    'recent_major_changes',
    'nursing_mothers',
    'pharmacokinetics_table',
    'general_precautions',
    'animal_pharmacology_and_or_toxicology',
    'references',
    'drug_abuse_and_dependence',
    'microbiology',
    'teratogenic_effects',
    'controlled_substance',
    'abuse',
    'drug_and_or_laboratory_test_interactions',
    'dependence',
    'keep_out_of_reach_of_children',
    'risks',
    'active_ingredient',
    'questions',
    'warnings_table',
    'purpose',
    'stop_use',
    'pregnancy_or_breast_feeding',
    'information_for_patients_table',
    'use_in_specific_populations_table',
    'nonteratogenic_effects',
    'patient_medication_information',
    'pharmacogenomics',
    'precautions_table',
    'microbiology_table',
    'pediatric_use_table',
    'patient_medication_information_table',
    'contraindications_table',
    'general_precautions_table',
    'other_safety_information',
    'mechanism_of_action_table',
    'safe_handling_warning',
    'pregnancy_table',
    'overdosage_table',
    'geriatric_use_table',
    'drug_abuse_and_dependence_table',
    'nonclinical_toxicology_table',
    'animal_pharmacology_and_or_toxicology_table',
    'abuse_table',
    'ask_doctor',
    'ask_doctor_or_pharmacist',
    'references_table',
    'carcinogenesis_and_mutagenesis_and_impairment_of_fertility_table',
    'troubleshooting',
    'components',
    'user_safety_warnings'
]

if __name__ == "__main__":
    spls = "/home/tom/Bureau/phd/mistral_training/hackaton_medi/fda_data/result/deduplicated_selection.json"
    spls = read_json(spls)


    drugs = []
    for k, v in spls.items():
        #if k not in set([str(x).upper() for x in ema['fda_match'].values]):
        line = {'name': k, 'fda_text': get_fda_text(k, spls, fields_fda, rag_mode=True), 'database': ['FDA'], 'fda_name': k}
        drugs.append(line)
    drugs = pd.DataFrame(drugs)
    drugs.to_csv('/home/tom/Bureau/phd/mistral_training/hackaton_medi/fda_data/result/drugs_complete_rag.csv', index=False)



    drugs = []
    for k, v in spls.items():
        #if k not in set([str(x).upper() for x in ema['fda_match'].values]):
            line = {'name': k, 'fda_text': get_fda_text(k, spls, fields_fda, rag_mode=False), 'database': ['FDA'], 'fda_name': k}
            drugs.append(line)
    drugs = pd.DataFrame(drugs)
    drugs.to_csv('/home/tom/Bureau/phd/mistral_training/hackaton_medi/fda_data/result/drugs_complete.csv', index=False)

    cont_text_not_found = 0
    list_len = []
    for text in drugs['fda_text']:
        if len(text) == 0:
            print("Empty text found")
            cont_text_not_found += 1
        else:
            print("Text found")
            # number of words
            list_len.append(len(text.split()))

## plot the distribution of the number of words

import matplotlib
import importlib
importlib.reload(matplotlib)
import matplotlib.pyplot as plt
plt.hist(list_len, bins=50)
## save the figure
plt.savefig('distribution_words.png')
plt.show()






import pandas as pd
import requests

# Load the CSV file
df = drugs
list_medicine_name = df['name'].tolist()
def extract_medicine_name(user_query):
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

def summarize_description(medicine_name, description):
    # Set up the API endpoint and headers
    api_endpoint = 'https://api.mistral.ai/summarize'
    headers = {
        'Authorization': open('api_key.txt').read().strip(),
        'Content-Type': 'application/json'
    }

    # Prepare the payload
    payload = {
        'medicine_name': medicine_name,
        'description': description
    }

    # Make the API request
    response = requests.post(api_endpoint, headers=headers, json=payload)

    # Return the summarized description
    return response.json().get('summary')

# Example usage
user_query = "digoxins?"
medicine_name = extract_medicine_name(user_query)
description = df[df['name'] == medicine_name]['fda_text'].values[0]
summary = summarize_description(medicine_name, description)
print(summary)


















