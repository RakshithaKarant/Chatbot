import os
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from CustomPrompt import CustomPrompt

class LLMHandler:
    def __init__(self, memory=None, persist_directory="chroma_storage"):
        load_dotenv()
        TOKEN = os.getenv("HF_TOKEN")
        if not TOKEN:
            raise ValueError("HF_TOKEN environment variable is not set.")
        
        # Initialize Hugging Face LLM
        self._repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
        model_kwargs = {"max_length": 128, "token": TOKEN}
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
        
        # Load and index CSV data with Chroma
        self.retriever = self._initialize_chroma_retriever("synthetic_airline_data.csv", persist_directory)
        self.prompt = CustomPrompt()

    def _initialize_chroma_retriever(self, csv_path, persist_directory):
        """Load CSV data and initialize a Chroma retriever."""
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Combine necessary fields into a descriptive string
        df['combined'] = df.apply(
            lambda row: f"Customer customer_id - {row['customer_id']} name - ({row['name']}): Flight {row['flight_id']} with {row['airline_name']} from {row['departure_city']} to {row['arrival_city']} on {row['departure_date']}. Booking ID: {row['booking_id']}.",
            axis=1
        )
        
        # Initialize embeddings
        embeddings = HuggingFaceHubEmbeddings()
        
        # Check if ChromaDB already exists for persistence
        if os.path.exists(persist_directory):
            # Load existing ChromaDB
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
        else:
            # Create ChromaDB and index data
            vector_store = Chroma.from_texts(
                texts=df.head(1000)['combined'].tolist(),
                embedding=embeddings,
                persist_directory=persist_directory
            )
            # Persist the vector store to disk
            vector_store.persist()
        
        return vector_store.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 results

    def handle_conversation(self, user_input: str):
        """Handle user input and generate a response using RAG."""
        # Retrieve relevant data
        relevant_context = self.retriever.get_relevant_documents(user_input)
        context_text = " ".join([doc.page_content for doc in relevant_context])
        
        # Create prompt with retrieved context
        formatted_prompt = self.prompt.create_prompt(user_input, additional_context=context_text)
        
        # Get response from the conversation chain
        response = self.conversation_chain.predict(input=formatted_prompt[0].content)
        self.memory.save_context({"input": formatted_prompt[0].content}, {"outputs": response})
        
        # Parse the response
        output_dict = self.prompt.parse_response(response)
        return output_dict
