import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint # Updated import
from langchain.schema import HumanMessage, SystemMessage

class llm_Demo:
    def __init__(self):
        load_dotenv()
        TOKEN = os.environ["HF_TOKEN"]
        self._repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
        model_kwargs = {"max_length": 128, "token": TOKEN}
        self._llm = HuggingFaceEndpoint(
            repo_id=self._repo_id,
            temperature=0.5,
            model_kwargs=model_kwargs
        )

    def get_repo_details(self) -> str:
        return self._repo_id

    def get_llm(self) -> str:
        return self._llm

    def _set_template(self):
        template_s = """You are a {style1}. 
                Tell me {count} facts about {event_or_place}.
             """

        prompt_template = ChatPromptTemplate.from_template(template_s)

        self.user_messages = prompt_template.format_messages(
            style1="knowledgeable historian",
            count=5,
            event_or_place="Taj Mahal"
        )

    def initiate_chat(self):
        self._set_template()
        # Initialize chat model
        
        chat_model = ChatHuggingFace(llm=self._llm)
        # Invoke response
        print(self.user_messages)
        response = chat_model.invoke(self.user_messages)
        print(response.content)

if __name__ == "__main__":
    llm_demo = llm_Demo()
    llm_demo.initiate_chat()
