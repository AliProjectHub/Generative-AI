# Import streamlit for app dev
import streamlit as st
# Import transformer classes for generaiton
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
# Import torch for datatype attributes
import torch
# Import the prompt wrapper...but for llama index
from llama_index.core.prompts.prompts import SimpleInputPrompt
# Import the llama index HF Wrapper
#from llama_index.llms.huggingface import HuggingFaceLLM
# Bring in embeddings wrapper
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
# Bring in HF embeddings - need these to represent document chunks
#from llama_index.legacy.embeddings.huggingface import HuggingFaceEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# Bring in stuff to change service context
from llama_index.legacy import ServiceContext
from llama_index.legacy import set_global_service_context
# Import deps to load documents
from langchain.document_loaders import PyMuPDFLoader
from llama_index.core import VectorStoreIndex 
from pathlib import Path

# Set auth token variable from hugging face
auth_token = "your_auth_token"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=auth_token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=auth_token)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create a system prompt
system_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as
helpfully as possible, while being safe. Your answers should not include
any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain
why instead of answering something not correct. If you don't know the answer
to a question, please don't share false information.

Your goal is to provide answers relating to the financial performance of
the company.<</SYS>>
"""
# Throw together the query wrapper
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

llm = HuggingFaceLLM(context_window=4096,
                    max_new_tokens=256,
                    system_prompt=system_prompt,
                    query_wrapper_prompt=query_wrapper_prompt,
                    model=model,
                    tokenizer=tokenizer)

# Create and dl embeddings instance
embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# Create new service context instance
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)

# And set the service context
set_global_service_context(service_context)

# Create PDF Loader
loader = PyMuPDFLoader()

# Load documents
documents = loader.load(file_path=Path('/vol/fob-vol3/mi20/jaabousa/Dokumente/annualreport.pdf'), metadata=True)

service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)

index = VectorStoreIndex.from_documents(documents, embed_model=embeddings)
query_engine = index.as_query_engine(llm=llm)

# Create centered main title 
st.title('LLM Analytics Banker')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    response = query_engine.query(prompt)
    # ...and write it out to the screen
    st.write(response)

    # Display raw response object
    with st.expander('Response Object'):
        st.write(response)
    # Display source text
    with st.expander('Source Text'):
        st.write(response.get_formatted_sources())
