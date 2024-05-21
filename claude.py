import textwrap
import anthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from search_models import SearchModel


class Claude(SearchModel):
    def __init__(self, API_KEY:str, use_embed:bool) -> None:
        self.API_KEY = API_KEY
        self.client = anthropic.Anthropic(api_key=API_KEY,)
        self.embed_model = "None"
        self.model = "claude-3-opus-20240229"
        self.use_embed = use_embed
    
    
    def embed_doc(self, chunks:list):
        pass
    #     response = self.client.embeddings.create(
    #             model= self.embed_model,
    #             input= chunks,
    #         )

    #     embeddings = [embed.embedding for embed in response.data]
    #     vector_database = {i: np.array(embed) for i, embed in enumerate(embeddings)}

    #     return {"chunks":chunks, "doc_embeddings":embeddings, "vector_database":vector_database}

    def embed_query(self, query:str):
        pass
    #     response = self.client.embeddings.create(
    #             model= self.embed_model,
    #             input= query,
    #         )

    #     query_embedding = response.data[0].embedding
    #     return {"query_text": query, "query_embed":query_embedding}
    

    def query(self, pdf_text:str, query:str):
        if False:
            # print("search With Embeddings!")
            # parsed_text, response, documents = self.query_with_embed(pdf_text, query)
            # return parsed_text
            pass
        else:
            print("give whole document as context!")
            pdf_text = pdf_text.replace("'", "").replace('"', "").replace("\n", " ")
            prompt = textwrap.dedent("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
                Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
                However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
                strike a friendly and converstional tone. \
                If the passage is irrelevant to the answer, you may ignore it.
                QUESTION: '{query}'
                PASSAGE: '{relevant_passage}'

                    ANSWER:
                """).format(query=query, relevant_passage=pdf_text)

            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            


            return response.content[0].text



    # def query_with_embed(self, pdf_text:str, query:str):
    #     chunks = self.preprocess_doc(pdf_text)
    #     document_dict = self.embed_doc(chunks)
    #     query_dict = self.embed_query(query)
    #     top_chunks_after_retrieval = self.get_top_chunks(query_dict, document_dict)

    #     # retrieved documents
    #     documents = [
    #         {"title": "chunk 0", "snippet": top_chunks_after_retrieval[0]},
    #         {"title": "chunk 1", "snippet": top_chunks_after_retrieval[1]},
    #         {"title": "chunk 2", "snippet": top_chunks_after_retrieval[2]},
    #     ]
    #     role_content = textwrap.dedent("""You are a helpful and informative bot that answers questions using text from the reference passages included below. \
    #         Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
    #         However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
    #         strike a friendly and converstional tone. \
    #         for each passage If the passage is irrelevant to the answer (try to also cite each passage if it is relevant), you may ignore it.
    #         First PASSAGE: '{top_chunk}'
    #         Second PASSAGE: '{second_top_schunk}'
    #         Third PASSAGE: '{third_top_chunk}'

    #             ANSWER:
    #         """).format(query=query, top_chunk=top_chunks_after_retrieval[0],  second_top_schunk = top_chunks_after_retrieval[1], third_top_chunk = top_chunks_after_retrieval[2])

    #     # get model response
    #     input = [
    #         {"role": "system", "content": role_content},
    #         {"role": "user", "content": query},
    #     ]

    #     response = self.client.chat.completions.create(
    #         model=self.model,
    #         messages=input,
    #         temperature=0
    #     )
    #     parsed_text = response.choices[0].message.content

    #     # parsed_text = self.insert_citations_in_order(response.text, response.citations, documents)

    #     return parsed_text, response, documents

    # def insert_citations_in_order(self, text:str, citations:list, documents:list):
    #     """
    #     A helper function to pretty print citations.
    #     """
    #     offset = 0
    #     document_id_to_number = {}
    #     citation_number = 0
    #     modified_citations = []

    #     # Process citations, assigning numbers based on unique document_ids
    #     if citations:
    #         for citation in citations:
    #             citation_numbers = []
    #             for document_id in sorted(citation["document_ids"]):
    #                 if document_id not in document_id_to_number:
    #                     citation_number += 1  # Increment for a new document_id
    #                     document_id_to_number[document_id] = citation_number
    #                 citation_numbers.append(document_id_to_number[document_id])

    #             # Adjust start/end with offset
    #             start, end = citation['start'] + offset, citation['end'] + offset
    #             placeholder = ''.join([f':blue[[{number}]]' for number in citation_numbers])
    #             # Bold the cited text and append the placeholder
    #             modification = f'**{text[start:end]}**{placeholder}'
    #             # Replace the cited text with its bolded version + placeholder
    #             text = text[:start] + modification + text[end:]
    #             # Update the offset for subsequent replacements
    #             offset += len(modification) - (end - start)

    #         # Prepare citations for listing at the bottom, ensuring unique document_ids are listed once
    #         unique_citations = {number: doc_id for doc_id, number in document_id_to_number.items()}
    #         citation_list = '\n'.join([f':blue[[{doc_id}]] source: "{documents[doc_id - 1]["snippet"]}" \n\n' for doc_id, number in sorted(unique_citations.items(), key=lambda item: item[1])])
    #         text_with_citations = f'{text} \n\n ------------------------------ Sources ------------------------------ \n\n {citation_list}'
    #     else:
    #         text_with_citations = text

    #     return text_with_citations  
