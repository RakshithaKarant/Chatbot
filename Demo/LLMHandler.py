import os
import re
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from RAG import Retrieval
from CustomPrompt import CustomPrompt
import time

class LLMHandler:
    def __init__(self, memory: ConversationBufferMemory = None):
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN environment variable is not set.")
        
        # Initialize Hugging Face LLM
        self._repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
        model_kwargs = {"max_length": 512, "token": token}
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
        
        # Initialize prompt handler
        self.prompt = CustomPrompt()

    def is_relevant(self, user_input: str) -> bool:
        """Check if the user input is relevant for additional details retrieval."""
        patterns = [
            r'flight id', r'booking id', r'phone number', r'name of'
        ]
        return any(re.search(pattern, user_input, re.IGNORECASE) for pattern in patterns)

    def summarize_important_details(self, text: str, max_tokens: int = 20) -> str:
        """Summarize and retain only the important details from the text within a max token limit."""
        lines = text.split('\n')
        important_lines = []
        current_tokens = 0
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['customer id', 'flight id', 'booking id', 'phone number', 'name']):
                # Count tokens in line
                line_tokens = len(line.split())
                if current_tokens + line_tokens > max_tokens:
                    break
                important_lines.append(line)
                current_tokens += line_tokens
                
        return "\n".join(important_lines)

    def handle_conversation(self, user_input: str) -> dict:
        """Handle user input and generate a response using RAG."""
        
        context_text = ""
        
        if self.is_relevant(user_input):
            # Start timer for retrieval
            start_time = time.time()
            context_text = Retrieval.get_additional_details(user_input)
            retrieval_time = time.time() - start_time
            print(f"Retrieval time: {retrieval_time:.4f} seconds")
        
        # Start timer for prompt creation
        start_time = time.time()
        formatted_prompt = self.prompt.create_prompt(user_input, additional_context=context_text)
        prompt_creation_time = time.time() - start_time
        print(f"Prompt creation time: {prompt_creation_time:.4f} seconds")
        
        # Start timer for LLM response generation
        start_time = time.time()
        response = self.conversation_chain.predict(input=formatted_prompt[0].content)
        llm_response_time = time.time() - start_time
        print(f"LLM response time: {llm_response_time:.4f} seconds")
        
        # Summarize important details before saving to memory
        summarized_input = self.summarize_important_details(formatted_prompt[0].content)
        summarized_response = self.summarize_important_details(response)
        
        # Start timer for saving context
        start_time = time.time()
        self.memory.save_context({"input": summarized_input}, {"outputs": summarized_response})
        context_saving_time = time.time() - start_time
        print(f"Context saving time: {context_saving_time:.4f} seconds")
        
        # Start timer for response parsing
        start_time = time.time()
        output_dict = self.prompt.parse_response(response)
        response_parsing_time = time.time() - start_time
        print(f"Response parsing time: {response_parsing_time:.4f} seconds")
        
        return output_dict
