import os
import re
import time
import logging
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from transformers import AutoTokenizer
import  Insights
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
    
        context_text=Insights.get_additional_context(user_input)
        
        # Create prompt
        start_time = time.time()
        raw_prompt = self.prompt.create_prompt(user_input, additional_context=context_text)
        truncated_prompt = self.truncate_to_max_tokens(raw_prompt[0].content)
        print(f"Prompt creation time: {time.time() - start_time:.4f} seconds")
        
        # Generate response
        start_time = time.time()
        response = self.conversation_chain.predict(input=truncated_prompt)
        truncated_response = self.truncate_to_max_tokens(response)
        print(f"LLM response time: {time.time() - start_time:.4f} seconds")
        
        # Summarize inputs and responses
        summarized_input = self.summarize_important_details(truncated_prompt)
        summarized_response = self.summarize_important_details(truncated_response)
        
        # Save context
        start_time = time.time()
        self.memory.save_context({"input": summarized_input}, {"outputs": summarized_response})
        print(f"Context saving time: {time.time() - start_time:.4f} seconds")
        
        # Parse response
        start_time = time.time()
        output_dict = self.prompt.parse_response(truncated_response)
        print(f"Response parsing time: {time.time() - start_time:.4f} seconds")
        
        return output_dict

# Set up logging to suppress debug messages
logging.basicConfig(level=logging.WARNING)  # Only show warnings and errors

if __name__ == "__main__":
    llm_handler = LLMHandler()
    tweets = [
        "What is the status of my flight booked in the name of Samantha Guerra and booking id is 41486539?",
        "What is my booking id?"
    ]

    for tweet in tweets:
        print(f"\nInput: {tweet}")
        result = llm_handler.handle_conversation(tweet)
        print("Output:", result)
