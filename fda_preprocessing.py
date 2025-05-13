
#%%
import json
import os
from glob import glob
from collections import defaultdict
import pandas as pd
from Levenshtein import distance
import numpy as np
from collections import Counter


def get_single_json(path):
    files = glob(os.path.join(path, "*.json"))
    single_json = []
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
            single_json.append(data['results'])
    return single_json


if __name__ == "__main__":
    dico = get_single_json("/home/tom/Bureau/phd/mistral_training/hackaton_medi/fda_data")
    dico = sum(dico, [])


    with open("/home/tom/Bureau/phd/mistral_training/hackaton_medi/fda_data/result/single_json.json", "w") as f:
         json.dump(dico, f)


    keys = []
    for d in dico:
        keys.extend(d.keys())



    """
    # Get all unii codes  Unique Ingredient Identifier
    unii = []
    c=0
    for x in dico:
        if 'openfda' in x:
            if 'unii' in x['openfda']:
                unii.append(x['openfda']['unii'])
            else:
                c+=1


    unii = Counter(sum(unii, []))
    len(unii)


    #Same Brand_name
    brand_name = []
    c=0
    X = []
    for x in dico:
        if 'openfda' in x:
            if 'brand_name' in x['openfda']:
                brand_name.append(x['openfda']['brand_name'])
            else:
                X.append(x)
    

    brand_name = Counter(sum(brand_name, []))
    len(brand_name)"""


    #Select the non-combination drugs, technique 1= len(unii)==1
    selection = [x for x in dico if ("unii" in x['openfda']) and (len(x['openfda']['unii'])==1)]
    len(selection)


    # From UniTox paper
    # We then grouped drugs by unique generic drug names and removed labels where the route of administration included topical, irrigational, or intradermal
    selection_by_drugname = defaultdict(list)
    for x in selection:
        if len(x['openfda']['generic_name']) == 1:
            selection_by_drugname[x['openfda']['generic_name'][0]].append(x)




    # Deduplicate - keep the latest version
    deduplicated_selection = {}
    for drug_name, entries in selection_by_drugname.items():
        try:
            # Sort entries by effective_time in descending order and keep the most recent one
            latest_entry = max(entries, key=lambda x: int(x.get('effective_time', '0')))
            deduplicated_selection[drug_name] = latest_entry
        except (ValueError, KeyError):
            # Skip entries with invalid effective_time format
            continue

    # %%
    not_included_routes = ['TOPICAL', 'IRRIGATION', "INTRADERMAL"]
    #selection_by_drugname = {k:v for k,v in selection_by_drugname.items() if ('route' not in v['openfda']) or (v['openfda']['route'][0] not in not_included_routes)}
    deduplicated_selection = {k:v for k,v in deduplicated_selection.items() if ('route' not in v['openfda']) or (v['openfda']['route'][0] not in not_included_routes)}

    with open("/home/tom/Bureau/phd/mistral_training/hackaton_medi/fda_data/result/deduplicated_selection.json", "w") as f:
         json.dump(deduplicated_selection, f)

    with open("/home/tom/Bureau/phd/mistral_training/hackaton_medi/fda_data/result/deduplicated_selection.json", "r") as f:
        deduplicated_selection = json.load(f)

    # %%