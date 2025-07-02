# Importing necessary libraries
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

from sympy.polys.polyconfig import query

class DocumentSimilarity:
    def __init__(self, documents: list, query: str) -> None:
        self.documents = documents
        self.query = query
        self.document_embeddings = None
        self.query_embedding = None
        self.model = None
        self.scores = None
    
    def model_creation(self):
        self.model = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2" # 90MB
        )

    def generate_embeddings(self):
        # Generating documents embeddings
        self.document_embeddings = self.model.embed_documents(texts = self.documents)

        # Generating query embedding
        self.query_embedding = np.array(self.model.embed_query(text = self.query)).reshape(1, -1)
    
    def similarity_score(self):
        # Calculating Cosine Similarity
        self.scores = cosine_similarity(self.document_embeddings, self.query_embedding)
    
    def similar_document(self):
        index, score = sorted(list(enumerate(self.scores)),key=lambda x:x[1])[-1]
        return self.documents[index]
    
    @classmethod
    def pipeline(class_name, documents: list, query: str):
        obj = class_name(documents, query)
        obj.model_creation()
        obj.generate_embeddings()
        obj.similarity_score()
        result = obj.similar_document()
        return obj, result

def main():
    # Storing model in D drive - Ingeneral, they were automatically saved in C drive
    os.environ["HF_HOME"] = 'D:/huggingface_cache'

    # Documents
    documents = [
        "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
        "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
        "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
        "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
        "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
    ]

    # Query
    query = input("Ask Anything: ")

    # Object creating and result
    obj, result = DocumentSimilarity.pipeline(
        documents = documents,
        query = query
    )
    # Printing
    print(f"Your Query: {query}")
    print(f"Similar Document: {result}")
    print(f"Similarity Scores: {obj.scores}")

if __name__ == "__main__":
    main()