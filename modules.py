from ragas.llms.base import LangchainLLMWrapper
from langchain.llms import OpenAI


class  LLMSRagAsm(LangchainLLMWrapper):
    def __init__(self,prompt):
        self.prompt = prompt

    def gen(self):
        ans = LangchainLLMWrapper(langchain_llm=OpenAI())

        return ans.generate_text(prompt=self.prompt,n=3)

