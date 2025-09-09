
from fastapi import FastAPI, HTTPException,UploadFile
from pydantic import BaseModel

import pandas as pd

import dspy
from dspy.dsp.utils import deduplicate
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.documents import Document
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS


app = FastAPI()

class Item(BaseModel):
    text: str = None 
    is_done: bool = False

lm = lm = dspy.LM('ollama_chat/mistral:7b', api_base='http://localhost:11434', api_key='',temperature=0.6)
dspy.configure(lm=lm)


def get_document(context):
    # Check if context is NaN or not a string
    if not isinstance(context, str) or pd.isna(context):
        print("Invalid context encountered; skipping.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=40
    )

    doc = [Document(page_content=context)]
    splitDocs = splitter.split_documents(doc)

    return splitDocs



def get_document_from_pdf(file_path):
    document_loader = PDFPlumberLoader(file_path)
    docs = document_loader.load()
    return docs[0].page_content

def create_db(docs):
    embedding = OllamaEmbeddings(model="nomic-embed-text:latest")
    vectorStore = FAISS.from_documents(docs,embedding)
    return vectorStore



def retrieve(inputs,vectordb):
    retriever = vectordb
    docs = retriever.max_marginal_relevance_search(inputs["question"], k=10, fetch_k=5, lambda_mult=0.5)
    return [doc.page_content for doc in docs]


# def embedd_chat_history(responce,vectorStore):
#     new_chat = Document(page_content=responce)
#     vectorStore.add_documents([new_chat])

class QApatient(dspy.Signature):
    """
    Assume the role of the patient described in the provided medical history.
    - Speak naturally and casually, reflecting a real person's mannerisms.
    - Use simple, everyday language; avoid medical jargon or technical terms.
    - Base your responses on the given medical history and current context.
    - Avoid suggesting any diagnoses or treatments.
    - Answer greetings in a natural, friendly manner.
    - If the doctor prescribes a treatment or test, briefly acknowledge it and end the conversation.
    - Keep responses concise, ideally between 10 to 20 words.
    """
    question = dspy.InputField(desc="Doctor's question")
    context = dspy.InputField(desc="may contain facts related to patient")
    answer = dspy.OutputField(desc="Patient's response")

class Query(dspy.Signature):
    """Write a optimal search query base on question and step_by_step_thoughts"""
    
    thoughts = dspy.InputField(desc="step_by_step_thoughts")
    question = dspy.InputField(desc="Question asked by the doctor")
    query = dspy.OutputField(desc="Query for search")
    
class COT_Patient_RAG(dspy.Module):
    def __init__(self):
        self.thoughts = dspy.ChainOfThought("question,context -> step_by_step_thoughts")
        self.respond = dspy.ChainOfThought(QApatient)
        self.query = dspy.ChainOfThought(Query)
        
    def forward(self,context,question):
        docs = get_document(context=context)
        global vectordb
        vectordb = create_db(docs=docs)
        context1 = retrieve({"question": "History and No History, Has and Not has sypmtoms"},vectordb=vectordb)
        thoughts = self.thoughts(context=context1,question=question).step_by_step_thoughts
        query = self.query(question=question,thoughts=thoughts)
        context2 = retrieve({"question": query.query},vectordb=vectordb)
        response =  self.respond(context=context2, question=question).answer
        return response
        


        
loaded_rag = COT_Patient_RAG()
loaded_rag.load('Optimized_RAG2.json')

@app.get("/")
def root():
    return {"DSPy":"Patient AI Home"}

FILE_PATH = 'document_store/pdfs/'
@app.post("/upload")
async def upload_file(file: UploadFile):
    contents = await file.read()
    file_path = FILE_PATH + file.filename
    with open(file_path, "wb") as f:
        f.write(contents)
    global docs
    docs = get_document_from_pdf(file_path=file_path)
    # global vectordb 
    # vectordb = create_db(docs)
    return {"message": "File received successfully"}


@app.post("/ask")
def responce(query:Item):
    try: 
        # loaded_rag = cot_rag
        # loaded_rag.load('Optimized_RAG2.json')
        responce = loaded_rag(context=docs,question=query)
        return {"ai_responce": responce}
    except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    