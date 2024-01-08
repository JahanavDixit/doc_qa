# Import necessary modules and define env variables

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import tempfile
from langchain.embeddings import SentenceTransformerEmbeddings
from chromadb.utils import embedding_functions
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import io
import chainlit as cl
import PyPDF2
from io import BytesIO


from dotenv import load_dotenv

# Load environment variables from .env file
#load_dotenv()

#OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")


# text_splitter and system template

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# system_template = """Use the following pieces of context to answer the users question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# ALWAYS return a "SOURCES" part in your answer.
# The "SOURCES" part should be a reference to the source of the document from which you got your answer.

# Example of your response should be:

# ```
# The answer is foo
# SOURCES: xyz
# ```

# Begin!
# ----------------
# {summaries}"""



# messages = [
#     SystemMessagePromptTemplate.from_template(system_template),
#     HumanMessagePromptTemplate.from_template("{question}"),
# ]
# prompt = ChatPromptTemplate.from_messages(messages)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain_type_kwargs = {"prompt": prompt}

@cl.on_chat_start
async def on_chat_start():

    # Sending an image with the local file path
    await cl.Message(content="Hello there, Welcome to Votum!").send()
    files = None

    # Wait for the user to upload a PDF file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            timeout=180,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()
    print(type(file))
    # Read the PDF file
    pdf_stream = BytesIO(file.content)
    print(type(pdf_stream))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_stream.read())
        temp_file_path = temp_file.name
    loader = PyPDFLoader(temp_file_path)
    pages = loader.load_and_split()

# Create a Chroma vector store
    embeddings = HuggingFaceEmbeddings()
    faiss_index = FAISS.from_documents(pages, embeddings)

# Clean up the temporary file
    os.remove(temp_file_path)    
    pdf = PyPDF2.PdfReader(pdf_stream)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

    # Split the text into chunks
    texts = text_splitter.split_text(pdf_text)

    # Create metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a chain that uses the Chroma vector store
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(model_name='TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ',
                openai_api_base='http://20.124.240.6:8083/v1',
                openai_api_key="EMPTY",
                streaming=False),
        chain_type="refine",
        retriever=faiss_index.as_retriever(),
    )
    

    # Save the metadata and texts in the user session
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message:str):
    message = message.content
    print("This" , message)
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])

    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []
    
    # Get the metadata and texts from the user session
    metadatas = cl.user_session.get("metadatas")
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get("texts")
    
    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip()
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = texts[index]
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()