from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from dotenv import load_dotenv
import os

# safety settings for Gemini
safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

GOOGLE_API_KEY = None
embeddings = None
model = None
vector_index = None
qa_chain = None



def get_api():
    load_dotenv()
    global GOOGLE_API_KEY
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def get_embedding():
    global embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

def load_docs():
    pdf_loader = PyPDFLoader("sample_docs/assessment.pdf")
    pages = pdf_loader.load_and_split() 
    return pages

def load_model():
    global model
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        safety_settings=safety_settings
    )

def create_vector_index(docs):
    global vector_index
    vector_index = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )
    vector_index = vector_index.as_retriever()

def create_qa_chain():
    global qa_chain
    template = """
Context:

The following is the context information the model has access to:
{context}

Question:

User's query: {question}

Instructions for the model:

If the context contains the information needed to answer the question, provide a clear and concise response based on the context without repeating the context verbatim.

Engage in basic conversational interactions such as greetings naturally and appropriately.
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=False,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )


def initial_load():
    get_api()
    docs = load_docs()
    get_embedding()
    create_vector_index(docs)
    load_model()
    create_qa_chain()
    return qa_chain

# Initialize QA chain once for the module
qa_chain = initial_load()

def query(question: str):
    if not question:
        question = "What is the attention mechanism?"
    
    result_dict = qa_chain.invoke({"query": question})
    return result_dict["result"]  # just return the answer string


if __name__ == "__main__":
    qa_chain = initial_load()
    try:
        while True:
            user_input = input("\nAsk a question about the PDF (or 'exit' to quit):\n> ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            answer = qa_chain.run(x)
            print(f"\nAnswer:\n{answer}\n")

    except KeyboardInterrupt:
        print("\nGoodbye!")


