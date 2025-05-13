
import sys
sys.path.append('/hackaton_medi')
import os
from mistralai import Mistral
import pandas as pd
import random
from tqdm import tqdm
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import numpy as np
from baseline.chunk import chunk_by_section
from baseline.utils import ASSISTANT_INSTRUCTION
import faiss
from pathlib import Path
# ########################################"
# DUPLICATION FROM create_prompt.py
# ########################################"

def get_text_embedding(input, client):
    embeddings_batch_response = client.embeddings.create(
        model="mistral-embed",
        inputs=input
    )
    return embeddings_batch_response.data[0].embedding

def rag_baseline(query, client,
                 model,
                 drugs_df,
                 chunk_size=60,
                 test_mode=True,
                 path_databased="/home/tom/Bureau/phd/mistral_training/hackaton_medi/rag_vector/IndexFlatL2"):



    chat_response = client.chat.complete(
        model= model,
        messages = [
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
    print(chat_response.choices[0].message.content)

    if chat_response.choices[0].message.content != 'I am not sure I understand the name of  drug, can you be more precise?':
        medicine_name = chat_response.choices[0].message.content.split(':')[0]
        print(f"medicine name: {medicine_name}")
        if medicine_name[0] == '"':
            medicine_name = medicine_name[1:]
        description_patient = chat_response.choices[0].message.content.split(':')[1]
    else:
        if test_mode:
            raise ValueError("The model was not able to extract the drug name from the query.")
        else:
            raise NotImplementedError("The model was not able to extract the drug name from the query.")

    ## try to load previous vector databased
    description = drugs_df[drugs_df['name'] == medicine_name.upper()]['fda_text'].values[0]
    chunks = chunk_by_section(description, chunk_size)
    try:
        index = faiss.read_index(str(Path(path_databased) / f"{medicine_name.upper()}.index"))
        print(f"Loading previous vector database for {medicine_name.upper()}")
    except:
        print("No index found, creating a new one.")
        text_embeddings  = []
        from tqdm import tqdm
        for chunk_index in tqdm(range(len(chunks))):
            chunk = chunks[chunk_index]
            text_embeddings.append(get_text_embedding(chunk, client))
            if chunk_index % 4 == 0:
                import time
                time.sleep(0.01) #Requests rate limit exceeded
        text_embeddings = np.array(text_embeddings)
        d = text_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(text_embeddings)
        faiss.write_index(index, str(Path(path_databased) / f"{medicine_name.upper()}.index"))


    question_embeddings = np.array([get_text_embedding(description_patient, client)])



    # retrive question
    D, I = index.search(question_embeddings, k=4)
    #print(I)
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
    #print(retrieved_chunk)

    chat_response = client.chat.complete(
        model= model,
        messages = [
            {
                "role": "assistant",
                "content":
                    f"You are a helpful assistant to a doctor who wants to prescribe {medicine_name} to a patient.\n\n"
                    f"The doctor describes the patient as follows: {description_patient}\n\n"
                    f"Here are extract of drug information sheet for {medicine_name}: {retrieved_chunk}\n\n"
                    f"First, summarise the information sheet.\n\n"
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
                "content": query
                ,
            },
        ]
    )

    print(chat_response.choices[0].message.content)

    recommendation_text = chat_response.choices[0].message.content.split("### Recommendation", 1)[-1]
    #print(recommendation_text)

    summary = chat_response.choices[0].message.content.split("### Recommendation", 1)[0]
    recommendation_text = chat_response.choices[0].message.content.split("### Recommendation", 1)[-1]
    Risk_score = recommendation_text.split("#### Risk score:")[-1].strip()

    #print(chat_response.choices[0].message.content)
    print(f"Risk_score pred by rag: {Risk_score}")
    return summary, recommendation_text, Risk_score, retrieved_chunk









if __name__ == "__main__":




    api_key = os.environ.get("MISTRAL_API_KEY")
    model = "ft:open-mistral-7b:86b3ddc9:20250511:aa021324"
    #model = "ministral-3b-latest"
    client = Mistral(api_key=api_key)

    unitox_df = pd.read_csv('/hackaton_medi/14042913/UniTox.csv')
    list_name_unitox = unitox_df['Generic Name'].tolist()
    print(unitox_df.columns)


    path_databased= "/hackaton_medi/rag_vector/IndexFlatL2"
    drugs_df = pd.read_csv \
        ('/hackaton_medi/fda_data/result/drugs_complete_rag.csv')

    query = "Should my patient takes 'Acyclovir' knowing he as a kidney failure?"
    chunk_size=60
    test_mode=True
    path_databased= "/hackaton_medi/rag_vector/IndexFlatL2"

    summary, recommendation_text, Risk_score = rag_baseline(query,
                 client,
                 model,
                chunk_size=60,
                 test_mode=True,
                 path_databased=path_databased,
                 drugs_df=drugs_df)














    drugs_df = pd.read_csv \
        ('/hackaton_medi/fda_data/result/drugs_complete_rag.csv')
    # Ensure the CSV file is in the same directory or provide the full path
    list_name_fda = drugs_df['name'].tolist()
    union_list = list(set(list_name_unitox).intersection(set(list_name_fda)))

    medicine_name = "ACYCLOVIR"
    description = drugs_df[drugs_df['name'] == medicine_name]['fda_text'].values[0]
    #print(description)

    # text chunking
    chunk_size = 60
    chunks = chunk_by_section(description, chunk_size)


    # text embedding

    text_embeddings  = []
    from tqdm import tqdm
    for chunk_index in tqdm(range(len(chunks))):
        chunk = chunks[chunk_index]
        text_embeddings.append(get_text_embedding(chunk) )
    text_embeddings = np.array(text_embeddings)
    index = faiss.read_index("text_embeddings.index")


    # FAISS index
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    faiss.write_index(index, f"{medicine_name}.index")
    index.add(text_embeddings)



    ## query
    question_embeddings = np.array([get_text_embedding(query)])
    question_embeddings.shape


    # retrive question
    D, I = index.search(question_embeddings, k=4)
    print(I)
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
    print(retrieved_chunk)




    chat_response = client.chat.complete(
        model= model,
        messages = [
            {
                "role": "assistant",
                "content":
                    f"You are a helpful assistant to a doctor who wants to prescribe {medicine_name} to a patient.\n\n"
                    f"The doctor describes the patient as follows: {description_patient}\n\n"
                    f"Here are extract of drug information sheet for {medicine_name}: {retrieved_chunk}\n\n"
                    f"First, summarise the information sheet.\n\n"
                    f'then, based on the patient description and the drug information, structure your response as follows:\n\n'
                    f"### Recommendation\n"
                    f"Give your professional opinion about prescribing this drug to the patient.\n\n"
                    f"Finally, assign a risk score and present it exactly as:\n"
                    f"#### Risk score: 0/1/2\n\n"
                    f"where 0 = appropriate for the patient, 1 = uncertain or potentially dangerous, and 2 = inappropriate for the patient."

            },
            {
                "role": "user",
                "content": query
                ,
            },
        ]
    )
    recommendation_text = chat_response.choices[0].message.content.split("### Recommendation", 1)[-1]
    print(recommendation_text)







    query = "Should my patient take  Acyclovir knowing he as a kidney failure?"
    chat_response = client.chat.complete(
        model= model,
        messages = [
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
    print(chat_response.choices[0].message.content)

    if chat_response.choices[0].message.content != 'I am not sure I understand the name of  drug, can you be more precise?':
        medicine_name = chat_response.choices[0].message.content.split(':')[0]
        description_patient = chat_response.choices[0].message.content.split(':')[1]


    question_embeddings = np.array([get_text_embedding(description_patient)])

    D, I = index.search(question_embeddings, k=10)
    print(I)
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
    print(retrieved_chunk)
    retrieved_chunk = " ".join(retrieved_chunk)


    chat_response = client.chat.complete(
        model= model,
        messages = [
            {
                "role": "assistant",
                "content":
                    f"You are a helpful assistant to a doctor who wants to prescribe {medicine_name} to a patient.\n\n"
                    f"The doctor describes the patient as follows: {description_patient}\n\n"
                    f"Here are extract of drug information sheet for {medicine_name}: {retrieved_chunk}\n\n"
                    f"First, summarise the information sheet.\n\n"
                    f'then, based on the patient description and the drug information, structure your response as follows:\n\n'
                    f"### Recommendation\n"
                    f"Give your professional opinion about prescribing this drug to the patient.\n\n"
                    f"Finally, assign a risk score and present it exactly as:\n"
                    f"#### Risk score: 0/1/2\n\n"
                    f"where 0 = appropriate for the patient, 1 = uncertain or potentially dangerous, and 2 = inappropriate for the patient."

            },
            {
                "role": "user",
                "content": query
                ,
            },
        ]
    )
    recommendation_text = chat_response.choices[0].message.content.split("### Recommendation", 1)[-1]
    summary = chat_response.choices[0].message.content.split("### Recommendation", 1)[0]
    recommendation_text = chat_response.choices[0].message.content.split("### Recommendation", 1)[-1]
    Risk_score = recommendation_text.split("#### Risk score:")[-1].strip()

    #print(chat_response.choices[0].message.content)



