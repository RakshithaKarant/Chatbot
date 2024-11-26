import os
import re
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from transformers import AutoTokenizer
from Insights import InsightDeriver, get_additional_context
from CustomPrompt import CustomPrompt

class LLMHandler:
    def __init__(self, memory: ConversationBufferMemory = None):
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN environment variable is not set.")
        
        # Initialize Hugging Face LLM
        self._repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
        model_kwargs = {"max_length": 500, "token": token}
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
        
        # Initialize tokenizer for managing token limits
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.max_tokens = 500  # Limit for the model
        self.prompt = CustomPrompt()
        self.insight_deriver = InsightDeriver()

    @staticmethod
    def is_relevant(user_input: str) -> bool:
        """Check if the user input is relevant for additional details retrieval."""
        patterns = [r'flight id', r'booking id', r'phone number', r'name of']
        return any(re.search(pattern, user_input, re.IGNORECASE) for pattern in patterns)

    def truncate_to_max_tokens(self, text: str, max_tokens: int = None) -> str:
        """Truncate the input text to ensure it stays within the token limit."""
        max_tokens = max_tokens or self.max_tokens
        tokens = self.tokenizer.encode(text, truncation=True, max_length=max_tokens)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def summarize_important_details(self, text: str, max_tokens: int = 500) -> str:
        """Summarize and retain only the important details from the text within a max token limit."""
        lines = text.split('\n')
        important_lines = []
        current_tokens = 0
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['customer id', 'flight id', 'booking id', 'phone number', 'name']):
                line_tokens = len(self.tokenizer.encode(line, truncation=True))
                if current_tokens + line_tokens > max_tokens:
                    break
                important_lines.append(line)
                current_tokens += line_tokens
                
        return "\n".join(important_lines)

    def handle_conversation(self, user_input: str) -> dict:
        """Handle user input and generate a response using RAG."""
    
        context_text = get_additional_context(user_input)
        
        # Create prompt
        raw_prompt = self.prompt.create_prompt(user_input, additional_context=context_text)
        truncated_prompt = self.truncate_to_max_tokens(raw_prompt[0].content)
    #Handle truncation to include both system and human messages
        
        # Summarize human input
        summarized_input = self.summarize_important_details(user_input)
        
        retry_count = 3
        output_dict = {"response": ""}
        
        while retry_count > 0:
            try:
                response = self.conversation_chain.predict(input=truncated_prompt)
                truncated_response = self.truncate_to_max_tokens(response)
                summarized_response = self.summarize_important_details(truncated_response)
                output_dict = self.prompt.parse_response(truncated_response)
                
                if "An unexpected error occurred" in output_dict["response"]:
                    raise Exception()
                
                retry_count = 0
                self.memory.save_context({"input": summarized_input}, {"outputs": summarized_response})
            except Exception as e:
                retry_count -= 1
        
        return output_dict

    def save_user_details(self, user_details):
        """Save user details in memory."""
        self.memory.save_context({"user_details": user_details}, {"outputs": ""})

