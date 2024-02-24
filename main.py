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
from modules import LLMSRagAsm,LLMSRag
from functions import Score,sent_tokenize,statements_prompt,nli_statements_generation,convert_json
from prompts import CONTEXT_PRECISION,CONTEXT_RELEVANCE
from openai import OpenAI




app = FastAPI()

l = {}
client = None
@app.on_event( "startup" )
async def start_up():
    global client
    client = OpenAI()


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


    
    obj = LLMSRag(prompt=QUESTION_GEN.format(answer=data["answer"],context=data["contexts"]))
    
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
        obj = LLMSRagAsm(prompt=i.prompt_str,client=client)
        json_respones_str.append(obj.gen())
    
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

    obj = LLMSRagAsm(prompt=context_relevancy_prompt.prompt_str,client=client)
    response = obj.gen()

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

#    #  ******************** Faithfullness ********************
@app.post("/faithfulness")
def faithfullness_score(item:Item):
    verdict_score_map = {"1": 1, "0": 0, "null": np.nan}
    question, answer,contexts = item.questions[0], item.answers[0],item.contexts[0]

    contexts = "\n".join(contexts)

    prompt = statements_prompt(question=question,answer=answer)

    statements = convert_json([LLMSRagAsm(prompt=prompt.prompt_str,client=client).gen()])[0]["statements"]

    prompt2 = nli_statements_generation(contexts=contexts,statements=statements)

    statements_verdicts = convert_json([LLMSRagAsm(prompt=prompt2.prompt_str,client=client).gen()])
    print(statements_verdicts)
    faithful_statements = sum(
    verdict_score_map.get(
        statement_with_validation.get("verdict", "").lower(), np.nan
    )
    for statement_with_validation in statements_verdicts[0].values()
)
    score= faithful_statements/len(statements_verdicts[0])
    l["faith_score"]  = score
    return score




@app.get("/eval/")
def  evaluate():
    return l






