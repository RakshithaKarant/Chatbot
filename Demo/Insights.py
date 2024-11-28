import pandas as pd
from sentence_transformers import SentenceTransformer
import regex as re
import faiss
import numpy as np
import os

class InsightDeriver:
    def __init__(self, filepath: str = "metadata.csv", vector_db_path: str = 'vector_db.index', threshold: float = 0.1):
        self.data = pd.read_csv(filepath)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_db_path = vector_db_path
        self.threshold = threshold
        self.insight_methods = {
            'total_bookings': self.total_bookings,
            'average_ticket_price': self.average_ticket_price,
            'most_popular_route': self.most_popular_route,
            'customer_age_distribution': self.customer_age_distribution,
            'class_distribution': self.class_distribution,
            'payment_method_distribution': self.payment_method_distribution,
            'airline_bookings': self.airline_bookings,
            'additional_services_distribution': self.additional_services_distribution,
            'most_frequent_flyer': self.most_frequent_flyer,
            'booking_trend': self.booking_trend,
            'revenue_per_airline': self.revenue_per_airline,
            'top_customers_by_spending': self.top_customers_by_spending,
            'customer_booking_frequency': self.customer_booking_frequency,
            'popular_departure_dates': self.popular_departure_dates,
            'popular_arrival_dates': self.popular_arrival_dates
        }
        self.insight_descriptions = list(self.insight_methods.keys())
        self.vector_db = self.load_or_build_vector_db(self.vector_db_path)

    def build_vector_db(self, descriptions):
        """Build a FAISS vector database for fast retrieval."""
        embeddings = self.model.encode(descriptions)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        faiss.write_index(index, self.vector_db_path)  # Save the index to the specified path
        return index

    def load_or_build_vector_db(self, vector_db_path):
        """Load the vector database if it exists, or build it if it doesn't."""
        if os.path.exists(vector_db_path):
            return faiss.read_index(vector_db_path)
        else:
            return self.build_vector_db(self.insight_descriptions)

    def query_vector_db(self, query, top_k=1):
        """Query the vector database to find the most similar description."""
        query_embedding = self.model.encode(query)
        distances, indices = self.vector_db.search(np.array([query_embedding]), top_k)
        
        # Check if the best match is above the threshold
        if distances[0][0] < self.threshold:
            return None
        return indices[0][0]

    def total_bookings(self):
        return len(self.data)

    def average_ticket_price(self):
        return self.data['ticket_price'].mean()

    def most_popular_route(self):
        route_counts = self.data.groupby(['departure_city', 'arrival_city']).size()
        most_popular = route_counts.idxmax()
        return most_popular, route_counts.max()

    def customer_age_distribution(self):
        return self.data['age'].describe()

    def class_distribution(self):
        return self.data['class'].value_counts()

    def payment_method_distribution(self):
        return self.data['payment_method'].value_counts()

    def airline_bookings(self):
        return self.data['airline_name'].value_counts()

    def additional_services_distribution(self):
        services = self.data['additional_services'].dropna().str.split(', ').explode()
        return services.value_counts()

    def user_details_by_booking_id(self, booking_id):
        user_details = self.data[self.data['booking_id'] == booking_id]
        if user_details.empty:
            return None
        return user_details.to_dict(orient='records')

    def status_summary(self):
        return self.data['status'].value_counts()

    def departure_time_distribution(self):
        self.data['departure_time'] = pd.to_datetime(self.data['departure_time'], format='%H:%M').dt.hour
        return self.data['departure_time'].value_counts().sort_index()

    def arrival_time_distribution(self):
        self.data['arrival_time'] = pd.to_datetime(self.data['arrival_time'], format='%H:%M').dt.hour
        return self.data['arrival_time'].value_counts().sort_index()
    
    # New methods for additional insights
    def most_frequent_flyer(self):
        frequent_flyer = self.data['customer_id'].value_counts().idxmax()
        return self.data[self.data['customer_id'] == frequent_flyer]['name'].iloc[0], frequent_flyer

    def booking_trend(self):
        self.data['booking_date'] = pd.to_datetime(self.data['booking_date'])
        return self.data.groupby(self.data['booking_date'].dt.to_period('M')).size()

    def revenue_per_airline(self):
        return self.data.groupby('airline_name')['ticket_price'].sum()

    def top_customers_by_spending(self, top_n=5):
        top_customers = self.data.groupby('customer_id')['ticket_price'].sum().nlargest(top_n)
        return self.data[self.data['customer_id'].isin(top_customers.index)][['customer_id', 'name', 'ticket_price']].drop_duplicates()

    def flight_duration_distribution(self):
        self.data['departure_time'] = pd.to_datetime(self.data['departure_time'], format='%H:%M')
        self.data['arrival_time'] = pd.to_datetime(self.data['arrival_time'], format='%H:%M')
        self.data['flight_duration'] = (self.data['arrival_time'] - self.data['departure_time']).dt.total_seconds() / 3600
        return self.data['flight_duration'].describe()

    def customer_booking_frequency(self):
        return self.data['customer_id'].value_counts().describe()

    def popular_departure_dates(self):
        self.data['departure_date'] = pd.to_datetime(self.data['departure_date'])
        return self.data['departure_date'].dt.date.value_counts().head()

    def popular_arrival_dates(self):
        return self.data['arrival_time'].dt.date.value_counts().head()

    def process_query(self, query: str):
        best_match_index = self.query_vector_db(query)
        if best_match_index is None:
            return {'error': 'No relevant insight found for the query'}
        best_match = self.insight_descriptions[best_match_index]
        return {best_match: self.insight_methods[best_match]()}  # Call the corresponding method

    @staticmethod
    def extract_first_number(query: str):
        """Extract the first occurrence of continuous digits in the string."""
        match = re.search(r'\d+', query)
        if match:
            return match.group()
        return None


def get_additional_context(query: str):
    insight_deriver = InsightDeriver()
    try:
        return insight_deriver.process_query(query)
    except:
        pass
    return None
