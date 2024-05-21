import streamlit as st
# from prompts import instructions_data
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
import streamlit.components.v1 as components
from io import StringIO
import pdfplumber
from command_r import CommandR
from gemini import GeminiFlash
from gpt import GPT
from claude import Claude

from st_audiorec import st_audiorec


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

Models = {
   "Model (R)":CommandR(st.secrets["COHERE_API_KEY"], False),
   "Model (R + Embed)":CommandR(st.secrets["COHERE_API_KEY"], True),
   "Model (F)":GeminiFlash(st.secrets["GOOGLE_API_KEY"], False),
   "Model (F + Embed)":GeminiFlash(st.secrets["GOOGLE_API_KEY"], True),
   "Model (F Audio)":GeminiFlash(st.secrets["GOOGLE_API_KEY"], False, use_audio=True),
   "Model (O)":GPT(st.secrets["OPENAI_API_KEY"], False),
   "Model (O + Embed)":GPT(st.secrets["OPENAI_API_KEY"], True),
   "Model (C)": Claude(st.secrets["ANTHROPIC_API_KEY"], False),

}



def chatbot(pdf_text, RAG_model):
  
  if RAG_model == "Model (F Audio)":
    # remove 
    wav_audio_data = st_audiorec()
    if wav_audio_data is not None:
        with st.chat_message("assistant"):
           with st.spinner("Thinking..."):
                model = Models[RAG_model]
                parsed_text = model.query_audio(pdf_text, wav_audio_data)
                st.write(parsed_text)
                # try:
                #     parsed_text = model.query(pdf_text=pdf_text, query=message)
                #     st.write(parsed_text)
                # except:
                #     parsed_text = "Server is starting...please try again in one minute"
                #     st.write(parsed_text)
        # st.audio(wav_audio_data, format='audio/wav')
    st.write("\n***\n")
    # remove
  else:
    if message := st.chat_input(key="input"):
        st.chat_message("user").write(message)
        st.session_state['chat_history'].append({"role": "user", "content": message})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                model = Models[RAG_model]
                parsed_text = model.query(pdf_text=pdf_text, query=message)
                try:
                    # response = output[0]["generated_text"]
                    parsed_text = model.query(pdf_text=pdf_text, query=message)
                    st.write(parsed_text)
                except:
                    parsed_text = "Server is starting...please try again in one minute"
                    st.write(parsed_text)

            message = {"role": "assistant", "content": parsed_text}
            st.session_state['chat_history'].append(message)
        st.write("\n***\n")


def chat():
    RAG_models_list = ["Model (F)", "Model (F + Embed)", "Model (O)", "Model (O + Embed)", "Model (C)",  "Model (R)", "Model (R + Embed)", "Model (F Audio)"]
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
                chatbot(pdf_text, RAG_model)
            

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