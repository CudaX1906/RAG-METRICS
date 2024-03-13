from fastapi import FastAPI,HTTPException
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
from prompts import CONTEXT_PRECISION,CONTEXT_RELEVANCE,QUESTION_GEN,FORMAT_CHECKER_CONVERTER,NLI_FORMAT_CHECKER_PROMPT
from openai import OpenAI
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    # if len(len(item.contexts[0]))==0 or len(item.questions)==0 or len(item.answers)==0:
    #     return 0
    prompt = QUESTION_GEN.format(answer = data["answer"],context=data["contexts"])
    
    resp = obj.gen(prompt.prompt_str)
    # print(resp)
    format_prompt = FORMAT_CHECKER_CONVERTER.format(input=resp)
    resp_for = obj.gen(format_prompt.prompt_str)
    resp_json = convert_json(response=[resp_for])
    
    àns_rel = Score(data["question"][0],resp_json)
    return àns_rel



#    #   *************** Context Precision ******************************

from fastapi import HTTPException, status

# ... (other imports and code)

@app.post("/contextprecision/")
def context_precision(items: Item):
    try:
        score = calculate_context_precision(items)
        return score
    except Exception as e:
        logger.error(f"Error in context_precision route: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )

def calculate_context_precision(items: Item):
    score = np.nan
    json_responses_str = []
    context_precision_prompt = []

    for c in items.contexts[0]:
        context_precision_prompt.append(CONTEXT_PRECISION.format(question=items.questions[0], context=c, answer=items.answers[0]))

    for i in context_precision_prompt:
        resp = obj.gen(i.prompt_str)
        logger.debug(f"API Response for context_precision: {resp}")
        json_responses_str.append(resp)

    json_responses = [json.loads(item) for item in json_responses_str]

    verdict_list = [int("1" == resp.get("verdict", "").strip()) if resp.get("verdict") else np.nan for resp in json_responses]
    denominator = sum(verdict_list) + 1e-10
    numerator = sum([(sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i] for i in range(len(verdict_list))])
    score = numerator / denominator

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
    prompt3 = NLI_FORMAT_CHECKER_PROMPT.format(input=response2)
    resp_format = obj.gen(prompt=prompt3.prompt_str)
    statements_verdicts = convert_json(response=[resp_format])
    print(statements_verdicts)
    # d = []
    
    #     d.append(statements_verdicts[0][i])
    faithful_statements = 0
    # print(d)
    try:
        for i in statements_verdicts[0]["answers"]:
            if i["verdict"]==1:
                faithful_statements = faithful_statements+1
    except TypeError as e:
        faithful_statements = 0
    
    # print(faithful_statements)
    if faithful_statements and len(statements_verdicts[0]["answers"]) > 0:
        score = faithful_statements / len(statements_verdicts[0]["answers"])
        # json_friendly_score = json.dumps(score, default=lambda x: str(x) if isinstance(x, np.float64) else x)
        l["faith_score"] = score
    else:
        l["faith_score"] = 0

   

    return l['faith_score']




@app.get("/eval/")
def  evaluate():
    return l