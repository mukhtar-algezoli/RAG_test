import streamlit as st
# from prompts import instructions_data
from clarifai.modules.css import ClarifaiStreamlitCSS
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import Clarifai
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
import streamlit.components.v1 as components
from io import StringIO
import pdfplumber
from CommandR import CommandR

import requests



# def query(payload): 
#  API_URL = "https://oa6kdk8gxzmfy79k.us-east-1.aws.endpoints.huggingface.cloud"

#  headers = {
#             "Accept" : "application/json", "Content-Type": "application/json" 
#             }
 
#  response = requests.post(API_URL, headers=headers, json=payload)
#  return response.json()

# st.set_page_config(layout="wide")



# def get_default_models():
#   # if 'DEFAULT_MODELS' not in st.secrets:
#   #   st.error("You need to set the default models in the secrets.")
#   #   st.stop()

#   # models_list = [x.strip() for x in st.secrets.DEFAULT_MODELS.split(",")]
#   models_list =  ["codegen2"]
#   authers = ["codegen2"]
#   apps = ["something"]
#   models_map = {}
#   select_map = {}
#   for i in range(len(models_list)):
#     m = models_list[i]
#     # id, rem = m.split(':')
#     # author, app = rem.split(';')
#     id = i
#     models_map[id] = {}
#     models_map[id]['author'] = authers[i]
#     models_map[id]['app'] = apps[i]
#     select_map[str(id)+' : '+ authers[i]] = i
#   return models_map, select_map


def show_previous_chats():
  # Display previous chat messages and store them into memory
  chat_list = []
  for message in st.session_state['chat_history']:
    with st.chat_message(message["role"]):
      if message["role"] == 'user':
        msg = HumanMessage(content=message["content"])
      else:
        msg = AIMessage(content=message["content"])
      chat_list.append(msg)
      st.write(message["content"])

  # conversation.memory.chat_memory = ChatMessageHistory(messages=chat_list)

def CommandR_insert_citations_in_order(text, citations, documents):
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




def chatbot(pdf_text):
  if message := st.chat_input(key="input"):
    st.chat_message("user").write(message)
    st.session_state['chat_history'].append({"role": "user", "content": message})
    with st.chat_message("assistant"):
      with st.spinner("Thinking..."):
        model = CommandR(st.secrets["COHERE_API_KEY"])
        output, documents = model.query(pdf_text=pdf_text, query=message)

        try:
            # response = output[0]["generated_text"]
            response = CommandR_insert_citations_in_order(output.text, output.citations, documents)
            st.write(response)
        except:
            response = "Server is starting...please try again in one minute"
            st.write(response)

        # print(output)
        # response = output[0]["generated_text"]
        # st.code(response, line_numbers=True)

        # st.write(response)
        # st.write(f":blue[{response}]")
        # st.write("This is :blue[test]")
        message = {"role": "assistant", "content": response}
        st.session_state['chat_history'].append(message)
    st.write("\n***\n")


def chat():
    RAG_models_list = ["Model Embed"]
    uploaded_file = st.file_uploader("Choose a file", type="pdf")


    if uploaded_file is not None:
            pdf_text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                pages = pdf.pages
                for p in pages:
                    pdf_text += p.extract_text() + "\n"
                   
            RAG_model = st.selectbox(
                'Select a Model',
                options=RAG_models_list,
                index=(RAG_models_list.index(st.session_state['RAG_model']) if 'Model Embed' in st.session_state else 0)
            )

            

            # Save the chosen option into the session state
            st.session_state['RAG_model'] = RAG_model

            # if st.session_state['RAG_model'] != "Select a Model":

            with open('styles.css') as f:
                st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)

            if 'RAG_model' not in st.session_state.keys():
                RAG_model = st.selectbox(label="Select an LLM", options= RAG_models_list)
                st.session_state['RAG_model'] = RAG_model


            # Access instruction by key
            # instruction = instructions_data[st.session_state['chosen_instruction_key']]['instruction']

            template = f"""{"Ask me something about your document and I will try to respond!"} + {{chat_history}}
            Human: {{input}}
            AI Assistant:"""

            prompt = PromptTemplate(template=template, input_variables=["chat_history", "input"])



            # Initialize the bot's first message only after LLM was chosen
            if "RAG_model" in st.session_state.keys() and "chat_history" not in st.session_state.keys():
                with st.spinner("Chatbot is initializing..."):
                    # initial_message = conversation.predict(input='', chat_history=[])
                    initial_message = "Ask me something about your document and I will try to respond!"
                    st.session_state['chat_history'] = [{"role": "assistant", "content": initial_message}]

            if "RAG_model" in st.session_state.keys():
                show_previous_chats()
                chatbot(pdf_text)

            st.markdown(
                """
            <style>
            .streamlit-chat.message-container .content p {
                white-space: pre-wrap !important;
                word-wrap: break-word !important;
                overflow-wrap: break-word !important;
            }
            .output {
                white-space: pre-wrap !important;
                }
            </style>
            """,
                unsafe_allow_html=True,
            )

chat()