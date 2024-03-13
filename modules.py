from ragas.llms.base import LangchainLLMWrapper
from langchain_openai.llms import OpenAI


class LLMSRagAsm():
    def __init__(self, client):
        self.client = client  

    def gen(self,prompt):
        completion = self.client.chat.completions.create(model="gpt-4-0125-preview", messages=[{"role": "user", "content": prompt}],response_format={ "type": "json_object" },temperature = 0)
        return completion.choices[0].message.content





class  LLMSRag(LangchainLLMWrapper):
    def __init__(self,prompt):
        self.prompt = prompt

    def gen(self):
        ans = LangchainLLMWrapper(langchain_llm=OpenAI(temperature=0))

        return ans.generate_text(prompt=self.prompt,n=3)