
from fastapi import FastAPI, HTTPException,UploadFile
from pydantic import BaseModel
import dspy
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from typing import List, Optional


app = FastAPI()

class Item(BaseModel):
    text: str = None 
    is_done: bool = False

dspy.configure(lm=dspy.LM('ollama_chat/gemma3:12b', api_base='http://localhost:11434', api_key='',temperature=0.6))

def get_document_from_pdf(file_path):
    document_loader = PyPDFLoader(file_path)
    docs = document_loader.load()
    return docs[0].page_content


def retrieve(inputs,vectorStore):
    retriever = vectorStore
    docs = retriever.max_marginal_relevance_search(inputs["question"], k=10, fetch_k=5, lambda_mult=0.5)
    return [doc.page_content for doc in docs]

global history
history = dspy.History(messages=[])


    
# ========== QAPatient Signature ==========
class QAPatient(dspy.Signature):
    """
    Role: Simulate a realistic patient in a doctor-patient conversation.

    Instructions:
        1. Respond only to the doctor's question using the given context and retrieved history.
        2. Do NOT add extra details not present in the context or history.
        3. Doctor’s question will be in Hinglish.
        4. Patient’s response must be in Hindi (Devanagari script), not in Hinglish or English.
        5. Keep responses concise (10–20 words).
        6. Maintain consistency with patient profile, symptoms, and tone.
    Critical:
        - STRICTLY follow context and retrieved history.
        - NEVER invent or assume any medical details.
        - Ensure output is only in Hindi (Devanagari script).
    """
    question: str = dspy.InputField(description="Doctor's question in Hinglish")
    context: str = dspy.InputField(description="Patient’s medical history and current condition")
    retrieved_history: Optional[str] = dspy.InputField(description="Relevant conversation history snippets")
    answer: str = dspy.OutputField(description="Patient’s response in Hindi (Devanagari script), 10–20 words")


# ========== HistoryRetriever Signature ==========
class HistoryRetriever(dspy.Signature):
    """
    Role: Retrieve the most relevant parts of the conversation history to help answer the doctor’s question.

    Instructions:
        1. Understand the doctor’s question within the current medical context.
        2. Retrieve only relevant past exchanges helpful to form the patient’s next response.
        3. Rank all retrieved history items on a scale of 1–10 for relevance.
        4. Return the top 3 ranked history items as a single string summary.
        5. Be concise and remove unnecessary details.
    """
    question: str = dspy.InputField(description="Doctor's question in Hinglish")
    context: str = dspy.InputField(description="Patient’s medical history and condition")
    history: Optional[dspy.History] = dspy.InputField(description="Complete conversation history object")
    retrieved_history: str = dspy.OutputField(description="Top 3 most relevant conversation history entries (ranked and summarized)")


# ========== ResponseJudge Signature ==========
class ResponseJudge(dspy.Signature):
    """
    Role: Evaluate the quality and accuracy of the patient’s response.

    Evaluation Criteria:
        1. Does the patient’s response directly answer the doctor’s question?
        2. Is it consistent with the provided context and medical history?
        3. Does it align with retrieved conversation history?
        4. Is it written entirely in Hindi (Devanagari script)?
        5. Is the response tone appropriate for a patient (not robotic or overly formal)?

    Output:
        - Assign a score between 1 to 5 (1 = poor, 5 = excellent)
    """
    question: str = dspy.InputField(description="Doctor's question in Hinglish")
    response: str = dspy.InputField(description="Patient’s generated response in Hindi (Devanagari script)")
    context: str = dspy.InputField(description="Patient’s medical history and context")
    retrieved_history: Optional[str] = dspy.InputField(description="Retrieved conversation history used for response generation")
    response_rating: int = dspy.OutputField(description="Quality rating between 1 (poor) and 5 (excellent)", ge=1, le=5)


# ========== FailCase Signature ==========
class FailCase(dspy.Signature):
    """
    Role: Provide a fallback patient response when confidence or rating is too low.

    Output:
        - A short Hindi sentence politely saying the patient is unable to answer.
    Example Output:
        "मुझे इस सवाल का जवाब ठीक से नहीं पता डॉक्टर साहब।"
    """
    fail_response: str = dspy.OutputField(description="Fallback response in Hindi (Devanagari script)")


class RAGmodule(dspy.Module):
    def __init__(self):
        self.response = dspy.Predict(QAPatient)
        self.response_rating = dspy.ChainOfThought(ResponseJudge)
        self.retrieve_history = dspy.ChainOfThought(HistoryRetriever)
        self.response_failure = dspy.Predict(FailCase)
    def forward(self,query:str):
        context = docs
        retrieved_history = self.retrieve_history(question=query,context=context,history=history)
        retrieved_history = retrieved_history.retrieved_history
        response = self.response(question=query,context=context,retrieved_history=retrieved_history)
        responce_rating = self.response_rating(question=query,response=response.answer,context=context,retrieved_history=retrieved_history)
        if responce_rating.response_rating<4:
            fail_response = self.response_failure().fail_response
            return fail_response
        history.messages.append({"question": query, "Patient":response.answer})
        return response.answer
        


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
    return {"message": "File received successfully"}


@app.post("/ask")
def responce(query:Item):
    try: 
        cot_rag = RAGmodule()
        responce = cot_rag(query=query.text)
        print("Responce ",responce)
        print("History: ",history.messages)
        return {"ai_responce": responce}
    except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    