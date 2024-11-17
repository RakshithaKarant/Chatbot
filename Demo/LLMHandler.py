import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint  # For the endpoint connection
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from CustomPrompt import CustomPrompt

class LLMHandler:
    def __init__(self, memory=None):
        load_dotenv()
        TOKEN = os.getenv("HF_TOKEN")
        if not TOKEN:
            raise ValueError("HF_TOKEN environment variable is not set.")
        
        self._repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
        model_kwargs = {"max_length": 128, "token": TOKEN}
        
        # Initialize Hugging Face endpoint
        self.llm = HuggingFaceEndpoint(
            repo_id=self._repo_id,
            temperature=0.5,
            model_kwargs=model_kwargs
        )
        
        # Use provided memory or create a new one
        self.memory = memory or ConversationBufferMemory()
        self.conversation_chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=False  
        )
        
        self.prompt = CustomPrompt()

    def handle_conversation(self, user_input: str):
        # Create formatted prompt messages
        formatted_prompt = self.prompt.create_prompt(user_input)
        # Get response from the conversation chain
        response = self.conversation_chain.predict(input=formatted_prompt[0].content)
        self.memory.save_context({"input": formatted_prompt[0].content}, {"outputs": response})
        # Parse the response
        output_dict = self.prompt.parse_response(response)
        return output_dict
