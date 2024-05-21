import cohere
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from search_models import SearchModel


class CommandR(SearchModel):
    def __init__(self, API_KEY:str, use_embed:bool, use_audio=False) -> None:
        self.API_KEY = API_KEY
        self.co = cohere.Client(self.API_KEY)
        self.embed_model = "embed-english-v3.0"
        self.model = "command-r"
        self.use_embed = use_embed
    
    
    def embed_doc(self, chunks:list):
        response = self.co.embed(
            texts= chunks,
            model= self.embed_model,
            input_type="search_document",
            embedding_types=['float']
        )
        embeddings = response.embeddings.float
        vector_database = {i: np.array(embedding) for i, embedding in enumerate(embeddings)}

        return {"chunks":chunks, "doc_embeddings":embeddings, "vector_database":vector_database}

    def embed_query(self, query:str):
        response = self.co.embed(
            texts=[query],
            model= self.embed_model,
            input_type="search_query",
            embedding_types=['float']
        )
        query_embedding = response.embeddings.float[0]
        return {"query_text": query, "query_embed":query_embedding}
    

    def query(self, pdf_text:str, query:str):
        if self.use_embed:
            print("search With Embeddings!")
            parsed_text, response, documents = self.query_with_embed(pdf_text, query)
            return parsed_text
        else:
            print("give whole document as context!")
            preamble = """
            ## Task & Context
            You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

            ## Style Guide
            Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.
            """
            base_prompt = "Based on the passage above, answer the following question:"
            new_prompt = '\n'.join(pdf_text) + '\n\n' + base_prompt + '\n' + query + '\n'
            response = self.co.chat(
                message=new_prompt,
                preamble=preamble,
                model=self.model,
                temperature=0.3
            )
            return response.text



    def query_with_embed(self, pdf_text:str, query:str):
        chunks = self.preprocess_doc(pdf_text)
        document_dict = self.embed_doc(chunks)
        query_dict = self.embed_query(query)
        top_chunks_after_retrieval = self.get_top_chunks(query_dict, document_dict)

        response = self.co.rerank(
            query=query_dict["query_text"],
            documents=top_chunks_after_retrieval,
            top_n=3,
            model="rerank-english-v2.0",
        )

        top_chunks_after_rerank = [result.document['text'] for result in response]

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

        parsed_text = self.insert_citations_in_order(response.text, response.citations, documents)

        return parsed_text, response, documents

    def insert_citations_in_order(self, text:str, citations:list, documents:list):
        """
        A helper function to pretty print citations.
        """
        offset = 0
        document_id_to_number = {}
        citation_number = 0
        modified_citations = []

        # Process citations, assigning numbers based on unique document_ids
        if citations:
            for citation in citations:
                citation_numbers = []
                for document_id in sorted(citation["document_ids"]):
                    if document_id not in document_id_to_number:
                        citation_number += 1  # Increment for a new document_id
                        document_id_to_number[document_id] = citation_number
                    citation_numbers.append(document_id_to_number[document_id])

                # Adjust start/end with offset
                start, end = citation['start'] + offset, citation['end'] + offset
                placeholder = ''.join([f':blue[[{number}]]' for number in citation_numbers])
                # Bold the cited text and append the placeholder
                modification = f'**{text[start:end]}**{placeholder}'
                # Replace the cited text with its bolded version + placeholder
                text = text[:start] + modification + text[end:]
                # Update the offset for subsequent replacements
                offset += len(modification) - (end - start)

            # Prepare citations for listing at the bottom, ensuring unique document_ids are listed once
            unique_citations = {number: doc_id for doc_id, number in document_id_to_number.items()}
            citation_list = '\n'.join([f':blue[[{doc_id}]] source: "{documents[doc_id - 1]["snippet"]}" \n\n' for doc_id, number in sorted(unique_citations.items(), key=lambda item: item[1])])
            text_with_citations = f'{text} \n\n ------------------------------ Sources ------------------------------ \n\n {citation_list}'
        else:
            text_with_citations = text

        return text_with_citations  
