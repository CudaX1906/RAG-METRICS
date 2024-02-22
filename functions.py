from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np
import json

def calculate_similarity(question,generated_questions):
    embeddings = HuggingFaceEmbeddings(model_name ="BAAI/bge-base-en")
    question_vec = np.asarray(embeddings.embed_query(question)).reshape(1,-1)
    gen_question_vec = np.asarray(embeddings.embed_documents(generated_questions))
    norm = np.linalg.norm(question_vec,axis=1)*np.linalg.norm(gen_question_vec,axis=1)
    return (np.dot(gen_question_vec, question_vec.T).reshape(-1,)/norm)

def Score(question,generated_question):
    gen = []
    for i in generated_question:
        gen.append(i.text)
    print(gen)
    gen_questions = [json.loads(item)["question"] for item in gen]
    # committal = 
    committal = [True if json.loads(item)["noncommittal"] == "0" else False for item in gen]

    if len(committal)!=0:
        score = calculate_similarity(question,gen_questions).mean()
    else:
        score = 0
    return score


