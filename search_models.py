from abc import ABC, abstractmethod
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

class SearchModel(ABC):
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def preprocess_doc(self, pdf_text:str):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )

        chunks_ = text_splitter.create_documents([pdf_text])
        chunks = [c.page_content for c in chunks_]
        print(f"The text has been broken down in {len(chunks)} chunks.")
        return chunks
    
    @abstractmethod
    def embed_doc(self, chunks:list):
        pass

    @abstractmethod
    def embed_query(self, query:str):
        pass


    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def get_top_chunks(self, query_dict, document_dict):
        # Calculate similarity between the user question & each chunk
        similarities = [self.cosine_similarity(query_dict["query_embed"], chunk) for chunk in document_dict["doc_embeddings"]]

        # Get indices of the top 10 most similar chunks
        sorted_indices = np.argsort(similarities)[::-1]

        # Keep only the top 10 indices
        top_indices = sorted_indices[:10]

        # Retrieve the top 10 most similar chunks
        top_chunks_after_retrieval = [document_dict["chunks"][i] for i in top_indices]

        return top_chunks_after_retrieval
    

    @abstractmethod
    def query(self, pdf_text:str, query:str):
        pass

