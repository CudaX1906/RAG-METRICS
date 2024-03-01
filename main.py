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
    global client,obj,obj1
    client = OpenAI()
    obj = LLMSRagAsm(client=client)
    


@app.get("/")
def home():
    return  "Evaluation Framework"

#    #    ***************  Answer Relevancy Score **********************
@app.post("/answerrelevance/")
def ans_relevance(item:Item):
    
    data = {
        "question":item.questions,
        "answer":item.answers,
        "contexts": item.contexts,
    }


    
    obj1=LLMSRag(prompt=QUESTION_GEN.format(answer=data["answer"],context=data["contexts"]))
    
    gen = obj1.gen().generations[0]
    
    ans_rel = Score(data["question"][0],gen)
    l["AnswerRelevancy"] = ans_rel
    return ans_rel



#    #   *************** Context Precision ******************************

@app.post("/contextprecision/")
def  context_precision(items:Item):
    score = np.nan
    json_respones_str = []
    context_precision_prompt = [CONTEXT_PRECISION.format(question=items.questions[0],context=c,answer=items.answers[0]) for c in items.contexts]
    for i in context_precision_prompt:
        resp = obj.gen(i.prompt_str)
        json_respones_str.append(resp)
    
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
    return score

#    #   *************** Context Relevancy *****************

@app.post("/contextrelevancy")
def  context_relevancy(items:Item):
    context_relevancy_prompt = CONTEXT_RELEVANCE.format(question=items.questions[0],context=items.contexts)

    response = obj.gen(prompt=context_relevancy_prompt.prompt_str)

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

    return l["context_relevancy_score"]

#    #  ******************** Faithfullness ********************
@app.post("/faithfulness")
def faithfullness_score(item: Item):
    verdict_score_map = {"1": 1, "0": 0, "Nil": np.nan}
    question, answer, contexts = item.questions[0], item.answers[0], item.contexts[0]

    contexts = "\n".join(contexts)

    prompt = statements_prompt(question=question, answer=answer)
    response1 = [obj.gen(prompt=prompt.prompt_str)]
    statements = convert_json(response=response1)[0]["statements"]

    prompt2 = nli_statements_generation(contexts=contexts, statements=statements)
    response2 = obj.gen(prompt=prompt2.prompt_str)
    statements_verdicts = convert_json(response=[response2])
    print(statements_verdicts)
    d = []
    for i in statements_verdicts[0]:
        d.append(statements_verdicts[0][i])
    faithful_statements = 0
    print(d)
    try:
        for i in d:
            if i["verdict"]=='1':
                faithful_statements = faithful_statements+1
    except TypeError as e:
        faithful_statements = 0
    
    print(faithful_statements)
    if faithful_statements and len(statements_verdicts[0]) > 0:
        score = faithful_statements / len(statements_verdicts[0])
        # json_friendly_score = json.dumps(score, default=lambda x: str(x) if isinstance(x, np.float64) else x)
        l["faith_score"] = score
    else:
        l["faith_score"] = 0

   

    return l['faith_score']




@app.get("/eval/")
def  evaluate():
    return l