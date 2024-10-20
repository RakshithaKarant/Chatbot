import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
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

        # Initialize conversation memory to keep track of the dialog
        self.memory = ConversationBufferMemory()

    def get_repo_details(self) -> str:
        return self._repo_id

    def get_llm(self) -> str:
        return self._llm

    def _set_template(self, query):
        """
        Sets the template based on the customer query using gating logic.
        The chatbot's responses change based on the context of the query.
        """
        if "haven’t received" in query or "late" in query:
            # Empathy chain for issues like delivery delays
            template_s = """
                You are a customer service agent using empathy. 
                The customer said: "{query}"
                Your response: "I understand how frustrating this might be. Let me help you as quickly as possible."
            """
        elif "broken" in query or "not working" in query:
            # Problem-solving chain for technical issues
            template_s = """
                You are a customer service agent helping to solve a problem.
                The customer said: "{query}"
                Your response: "I’m sorry you’re experiencing this issue. Let me guide you step-by-step to resolve it."
            """
        else:
            # Escalation chain for unresolved issues
            template_s = """
                You are a customer service agent handling escalations.
                The customer said: "{query}"
                Your response: "It looks like I’m unable to assist you fully at the moment. I’m escalating your issue to our support team."
            """

        prompt_template = ChatPromptTemplate.from_template(template_s)

        # Format the user message
        self.user_messages = prompt_template.format_messages(query=query)

    def initiate_chat(self, customer_query):
        # Save the customer query in memory
        # Retrieve past conversation from memory
        past_conversation = self.memory.load_memory_variables({})

        # Add context from memory to the chatbot
        self._set_template(customer_query)
        
        # Initialize chat model
        chat_model = ChatHuggingFace(llm=self._llm)


        # Invoke response
        response = chat_model.invoke(self.user_messages)

        # Save the chatbot's response in memory
        self.memory.save_context(inputs={"customer_query": customer_query},outputs= {"bot_response": response.content})

        # Print past conversations along with the latest response
        #print(f"Past Conversations: {past_conversation}")
        print(f"Bot Response: {response.content}")


if __name__ == "__main__":
    # Example customer queries to test the gated logic with memory
    query1 = "Hi, I am John. I haven’t received my order yet, and it’s been delayed for days."
    query2 = "My product is broken, and I can’t get it to work."
    query3 = "I want to cancel my order."

    llm_demo = llm_Demo()

    # Testing multiple queries to demonstrate memory
    print("Response for Query 1 (Empathy):")
    llm_demo.initiate_chat(query1)

    print("\nResponse for Query 2 (Problem-Solving):")
    llm_demo.initiate_chat(query2)

    print("\nResponse for Query 3 (Escalation):")
    llm_demo.initiate_chat(query3)
