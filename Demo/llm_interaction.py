import os
import logging
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint  # For the endpoint connection
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Set up logging to suppress debug messages
logging.basicConfig(level=logging.WARNING)  # Only show warnings and errors

# Define a custom class for handling the prompt and parsing
class CustomPrompt:
    def __init__(self):
        # Define response schemas for output parsing
        self.sentiment = ResponseSchema(
            name="sentiment",
            description="Analyze sentiment state if user is angry/sad/neutral/happy/excited."
        )
        self.response = ResponseSchema(
            name="response",
            description="Provide a professional response to the tweet within the travel domain."
        )
        self.output_parser = StructuredOutputParser.from_response_schemas(
            [self.sentiment, self.response]
        )

    def create_prompt(self, text: str):
        format_instructions = self.output_parser.get_format_instructions()
        review_template = """
        The following text is a tweet from a user.
        Extract the following information:

        - sentiment: Analyze if the user is angry, sad, neutral, happy, or excited.
        - response: Provide a professional response to the tweet within the travel domain.

        Text: {text}

        {format_instructions}
        """
        # Create and format the chat prompt
        prompt = ChatPromptTemplate.from_template(template=review_template)
        messages = prompt.format_messages(text=text, format_instructions=format_instructions)
        return messages

    def parse_response(self, response: str):
        return self.output_parser.parse(response)

# Define the main class to handle the LLM and conversation chain
class LLMHandler:
    def __init__(self):
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
        
        # Initialize conversation memory and chain without verbosity
        self.memory = ConversationBufferMemory()
        self.conversation_chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=False  # Suppress verbose output
        )
        
        # Initialize custom prompt for creating and parsing messages
        self.prompt = CustomPrompt()

    def handle_conversation(self, user_input: str):
        # Create formatted prompt messages
        formatted_prompt = self.prompt.create_prompt(user_input)
        
        # Get response from the conversation chain
        response = self.conversation_chain.predict(input=formatted_prompt[0].content)
        
        # Parse the response
        output_dict = self.prompt.parse_response(response)
        return output_dict

if __name__ == "__main__":
    llm_handler = LLMHandler()
    tweets = ["I won the match", "That's rude", "What an idiot", "hello world in python"]
    
    for tweet in tweets:
        print(f"\nInput: {tweet}")
        result = llm_handler.handle_conversation(tweet)
        print("Output:", result)
