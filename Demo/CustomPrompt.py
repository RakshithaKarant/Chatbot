
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


class CustomPrompt:
    def __init__(self):
        # output parsing
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