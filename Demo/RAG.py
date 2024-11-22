import pandas as pd
import re
import os

class DataProcessor:
    def __init__(self, filepath: str):
        self.data = pd.read_csv(filepath)
        self.data.set_index(['customer_id', 'name', 'flight_id', 'phone_number', 'booking_id'], inplace=True)

    def get_data(self) -> pd.DataFrame:
        return self.data

class VectorDatabase:
    def __init__(self):
        self.vector_db = None

    def set_metadata(self, data: pd.DataFrame) -> None:
        self.vector_db = data

    def save(self, metadata_path: str) -> None:
        self.vector_db.to_csv(metadata_path)

    def load(self, metadata_path: str) -> None:
        self.vector_db = pd.read_csv(metadata_path, index_col=['customer_id', 'name', 'flight_id', 'phone_number', 'booking_id'])

class QueryProcessor:
    @staticmethod
    def extract_filters(query: str) -> dict:
        filters = {}
        patterns = {
            'name': r'name of ([\w\s]+?)(?= and)',
            'flight_id': r'flight id is (\w+)+?(?= and)',
            'phone_number': r'phone number is (\d+)+?(?= and)',
            'booking_id': r'booking id is (\w+)+?(?= and)'
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                filters[key] = match.group(1).strip()

        return filters

    def retrieve(self, query: str, vector_db: VectorDatabase) -> list:
        filters = self.extract_filters(query)
        filtered_data = vector_db.vector_db

        # Apply filters using the index
        for key, value in filters.items():
            filtered_data = filtered_data.loc[filtered_data.index.get_level_values(key).str.contains(value, case=False, na=False)]

        if filtered_data.empty:
            return []

        results = filtered_data.reset_index().to_dict(orient='records')
        return results

    def generate_response(self, query: str, vector_db: VectorDatabase) -> str:
        results = self.retrieve(query, vector_db)
        if not results:
            return "No matching records found."

        response_lines = []
        for record in results:
            response_line = (
                f"Customer ID {record['customer_id']}: {record['name']} (phone: {record['phone_number']}) "
                f"booked a flight with ID {record['flight_id']} and booking ID {record['booking_id']}."
            )
            response_lines.append(response_line)
        
        return "\n".join(response_lines)
    
class Retrieval:
    @staticmethod
    def get_additional_details(query: str) -> str:
        metadata_path = "metadata.csv"

        vector_db = VectorDatabase()

        if os.path.exists(metadata_path):
            # Load the vector database if available
            vector_db.load(metadata_path)
        else:
            # Initialize objects and process data
            data_processor = DataProcessor("synthetic_airline_data.csv")
            data = data_processor.get_data()
            vector_db.set_metadata(data)
            # Save the vector database
            vector_db.save(metadata_path)

        # Query processing
        query_processor = QueryProcessor()
        return query_processor.generate_response(query, vector_db)
