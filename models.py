from typing import List ,Optional
from pydantic import BaseModel


class Item(BaseModel):
    questions:List[str]
    contexts: List[List[str]]
    answers: List[str]
    # ground_truths : List[List[str]]

# class Datasets(BaseModel):
#     items:Item

class DatasetInfo(BaseModel):
    features: List[str]
    num_rows: int


