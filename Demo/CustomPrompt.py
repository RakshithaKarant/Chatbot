from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import json

class CustomPrompt:
    def __init__(self):
        # Define response schemas for output parsing
        self.response = ResponseSchema(
            name="response",
            description="Respond to the user's query within the airlines travel domain."
        )
        self.output_parser = StructuredOutputParser.from_response_schemas(
            [self.response]
        )

    def create_prompt(self, text: str, additional_context: str = "") -> list:
        """Create a formatted prompt with additional context."""
        additional_context = additional_context or "No additional context available."
        format_instructions = self.output_parser.get_format_instructions()
        review_template = """
        The following text is a query from a user.
        
        Additional context: {additional_context}
        
        Extract the following information:
        - response: Respond to the user's query within the airlines travel domain.

        Text: {text}

        {format_instructions}
        """
        prompt = ChatPromptTemplate.from_template(template=review_template)
        messages = prompt.format_messages(
            text=text,
            additional_context=additional_context,
            format_instructions=format_instructions
        )
        return messages

    def parse_response(self, response: str):
        """Parse the LLM's response into a structured format."""
        try:
            response_=self.output_parser.parse(response)
            if type(response_) is str:
                return {
                "response": response_
            }
            return response_
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing structured output: {response}")
            return {
                "response": "An unexpected error occurred. Please try again with relevent question."
            }
