import cohere
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np


class CommandR:
    def __init__(self, COHERE_API_KEY) -> None:
        self.API_KEY = COHERE_API_KEY
        self.co = cohere.Client(self.API_KEY)
        self.embed_model = "embed-english-v3.0"
        self.model = "command-r"
    
    def preprocess_doc(self, pdf_text:str):
        # split the document into chunks and 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )

        chunks_ = text_splitter.create_documents([pdf_text])
        chunks = [c.page_content for c in chunks_]
        print(f"The text has been broken down in {len(chunks)} chunks.")

        response = self.co.embed(
            texts= chunks,
            model= self.embed_model,
            input_type="search_document",
            embedding_types=['float']
        )
        embeddings = response.embeddings.float
        vector_database = {i: np.array(embedding) for i, embedding in enumerate(embeddings)}

        return {"chunks":chunks, "doc_embeddings":embeddings, "vector_database":vector_database}

    def preprocess_query(self, query:str):
        response = self.co.embed(
            texts=[query],
            model= self.embed_model,
            input_type="search_query",
            embedding_types=['float']
        )
        query_embedding = response.embeddings.float[0]
        return {"query_text": query, "query_embed":query_embedding}
    
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

        response = self.co.rerank(
            query=query_dict["query_text"],
            documents=top_chunks_after_retrieval,
            top_n=3,
            model="rerank-english-v2.0",
        )

        top_chunks_after_rerank = [result.document['text'] for result in response]
        return top_chunks_after_rerank

    def query(self, pdf_text:str, query:str):

        document_dict = self.preprocess_doc(pdf_text)
        query_dict = self.preprocess_query(query)
        top_chunks_after_rerank = self.get_top_chunks(query_dict, document_dict)

        # preamble containing instructions about the task and the desired style for the output.
        preamble = """
        ## Task & Context
        You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

        ## Style Guide
        Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.
        """
        
        # retrieved documents
        documents = [
            {"title": "chunk 0", "snippet": top_chunks_after_rerank[0]},
            {"title": "chunk 1", "snippet": top_chunks_after_rerank[1]},
            {"title": "chunk 2", "snippet": top_chunks_after_rerank[2]},
        ]

        # get model response
        response = self.co.chat(
        message=query,
        documents=documents,
        preamble=preamble,
        model=self.model,
        temperature=0.3
        )

        return response, documents

