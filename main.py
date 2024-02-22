from fastapi import FastAPI
from typing import List,Dict
from models import Item ,DatasetInfo
import asyncio
import pickle
from ragas import evaluate
from datasets import Dataset
from langchain_openai import OpenAIEmbeddings
import numpy as np
import json
from ragas.metrics import (
    faithfulness,
)
from prompts import QUESTION_GEN
from modules import LLMSRagAsm
from functions import Score,sent_tokenize
from prompts import CONTEXT_PRECISION,CONTEXT_RELEVANCE





app = FastAPI()

l = {}
@app.get("/")
def home():
    return  "Evaluation Framework"

#    #    ***************  Answer Relevancy Score **********************
@app.post("/answerrelevace/")
def ans_relevance(item:Item):
    
    data = {
        "question":item.questions,
        "answer":item.answers,
        "contexts": item.contexts,
    }


    
    obj = LLMSRagAsm(prompt=QUESTION_GEN.format(answer=data["answer"],context=data["contexts"]))
    gen = obj.gen().generations[0]
    ans_rel = Score(data["question"][0],gen)
    l["AnswerRelevancy"] = ans_rel
    return "AnswerRelevance Score"



#    #   *************** Context Precision ******************************

@app.post("/contextprecision/")
def  context_precision(items:Item):
    score = np.nan
    json_respones_str = []
    context_precision_prompt = [CONTEXT_PRECISION.format(question=items.questions[0],context=c,answer=items.answers[0]) for c in items.contexts]
    for i in context_precision_prompt:
        obj = LLMSRagAsm(prompt=i)
        json_respones_str.append(obj.gen().generations[0][0].text)
    
    json_responses = [
    json.loads(item) for item in json_respones_str
]
    verdict_list = [
            int("1" == resp.get("verdict", "").strip())
            if resp.get("verdict")
            else np.nan
            for resp in json_responses
        ]
    denominator = sum(verdict_list) + 1e-10
    numerator = sum(
            [
                (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
                for i in range(len(verdict_list))
            ]
        )
    score = numerator / denominator
    l["context_precision_score"] = score
    return "Context Precision Score"

#    #   *************** Context Relevancy *****************

@app.post("/contextrelevancy")
def  context_relevancy(items:Item):
    context_relevancy_prompt = CONTEXT_RELEVANCE.format(question=items.questions[0],context=items.contexts)

    obj = LLMSRagAsm(prompt=context_relevancy_prompt)
    response = obj.gen().generations[0][0].text

    context = "\n".join(items.contexts[0])
    context_sents = sent_tokenize(context)
    indices = (
            sent_tokenize(response.strip())
            if response.lower() != "insufficient information."
            else []
        )
        # print(len(indices))
    if len(context_sents) == 0:
        l["context_relevancy_score"] =0
    else:
        l["context_relevancy_score"] = min(len(indices) / len(context_sents), 1)

    return "Context Relevancy Score"


@app.get("/eval/")
def  evaluate():
    return l






