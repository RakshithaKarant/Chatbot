from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import json

class CustomPrompt:
    def __init__(self):
        # Define response schemas for output parsing
        self.sentiment = ResponseSchema(
            name="sentiment",
            description="Analyze sentiment state if user is angry/sad/neutral/happy/excited."
        )
        self.response = ResponseSchema(
            name="response",
            description="Provide a professional response to the tweet within the travel domain along with additional_context if available"
        )
        self.output_parser = StructuredOutputParser.from_response_schemas(
            [self.sentiment, self.response]
        )

    def create_prompt(self, text: str, additional_context: str = "") -> list:
        """Create a formatted prompt with additional context."""
        additional_context = additional_context or "No additional context available."
        format_instructions = self.output_parser.get_format_instructions()
        review_template = """
        The following text is a tweet from a user.
        
        Additional context: {additional_context}
        
        Extract the following information:
        - sentiment: Analyze if the user is angry, sad, neutral, happy, or excited.
        - response: Provide a professional response to the tweet within the travel domain along with additional_context if available.

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

    def parse_response(self, response: str) -> dict:
        """Parse the LLM's response into a structured format."""
        try:
            parsed_output = self.output_parser.parse(response)
            return parsed_output
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            return {
                "error": "JSONDecodeError",
                "message": "The response could not be parsed as valid JSON.",
                "raw_response": response
            }
        except ValueError as e:
            print(f"Error parsing structured output: {e}")
            return {
                "error": "ValueError",
                "message": "The structured output could not be parsed correctly.",
                "raw_response": response
            }
