from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np
import json
import pysbd
from typing import List
from ragas.llms.json_load import json_loader
from prompts import LONG_FORM_ANSWER_PROMPT,NLI_STATEMENTS_MESSAGE
import requests
seg = pysbd.Segmenter(language="en", clean=False)

def calculate_similarity(question,generated_questions):
    embeddings = HuggingFaceEmbeddings(model_name ="BAAI/bge-base-en")
    question_vec = np.asarray(embeddings.embed_query(question)).reshape(1,-1)
    gen_question_vec = np.asarray(embeddings.embed_documents(generated_questions))
    norm = np.linalg.norm(question_vec,axis=1)*np.linalg.norm(gen_question_vec,axis=1)
    return (np.dot(gen_question_vec, question_vec.T).reshape(-1,)/norm)

def Score(question,generated_question):
    gen = []
    # for i in generated_question:
    #     gen.append(i.text)
    print(generated_question)
    gen_questions = []
    # gen_questions = [json.loads(item)[0]["question"] for item in generated_question]
    # # committal = 
    # committal = [True if json.loads(item)[0]["noncommittal"] == "0" else False for item in generated_question]
    committal = []
    
    for item in generated_question[0]["output"]:
        print(item)
        if item["noncommittal"] == 0:
            gen_questions.append(item["question"])
            committal.append(True)

    print(gen_questions)
    if len(committal)!=0:
        score = calculate_similarity(question,gen_questions).mean()
    else:
        score = 0
    return score

def sent_tokenize(text: str):
    sentences = seg.segment(text)
    assert isinstance(sentences, list)
    return sentences

def statements_prompt(question,answer):
    
    prompt = LONG_FORM_ANSWER_PROMPT.format(question=question,answer=answer)
    return prompt

def  nli_statements_generation(contexts,statements):
    
    context_str = "\n".join(contexts)

    if statements==[]:
        statements = ["Nill"]
    
    statements_str = "\n".join(
            [f"statement_{i+1}: {st}" for i, st in enumerate(statements)]
        )
    prompt = NLI_STATEMENTS_MESSAGE.format(
    context=context_str, statements=statements_str
        )
    
    return prompt

def convert_json(response):
    json_form = [
    json.loads(item) for item in response
]
    return json_form


def request_metric_api(api_url, data):
    try:
        response = requests.post(api_url, json=data)
        response.raise_for_status()
        result = response.json()
        return result
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error calling API: {str(e)}")