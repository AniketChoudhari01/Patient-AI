
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

dspy.configure(lm=dspy.LM('ollama_chat/gemma3:4b', api_base='http://localhost:11434', api_key='',temperature=0.6))

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


    
# ========== QAPatient Signature (Base) ==========
class QAPatient(dspy.Signature):
    """
    Persona: You are a patient. Answer the doctor's question based *only* on the context.

    **CRITICAL Rules:**
    1.  **Hindi ONLY:** Respond **STRICTLY** in pure Hindi (Devanagari).
    2.  **NO Hinglish/Jargon:** Do NOT use Roman characters, English words, or transliterated
        jargon (e.g., `मूकोपुरulent`). Use simple Hindi descriptions.
    3.  **NO Questions:** You **MUST NOT** ask questions back.
    4.  **Context ONLY:** Base your answer *only* on the `context`. Do NOT invent symptoms.
    5.  **Direct & Concise:** Answer *only* the `question` in 1-2 sentences.
    """
    question: str = dspy.InputField(description="Doctor's question (Hindi).")
    context: str = dspy.InputField(description="Patient’s medical history.")
    retrieved_history: Optional[str] = dspy.InputField(description="Relevant past conversation.")
    answer: str = dspy.OutputField(description="Patient’s concise, pure Hindi response.")


# ========== HistoryRetriever Signature ==========
class HistoryRetriever(dspy.Signature):
    """
    Role: Extract relevant conversation history snippets for the current question.

    Instructions:
    1.  Focus only on context relevant to the current `question`.
    2.  Summarize concisely (1-2 lines).
    3.  **CRITICAL:** If no history is relevant, output an empty string: ''
    """
    question: str = dspy.InputField(description="Doctor's current question.")
    context: str = dspy.InputField(description="Patient’s medical history.")
    history: str = dspy.InputField(description="Full conversation history.")
    retrieved_history: str = dspy.OutputField(description="Concise summary of relevant history.")


# ========== ResponseJudge Signature ==========
class ResponseJudge(dspy.Signature):
    """
    Role: Act as a strict judge. Output a single integer score (1-5).

    **Judging Criteria (MUST follow):**
    1.  **Rating 1 (Fail):** Uses ANY Roman/English/Jargon (e.g., `मूकोपुरulent`).
    2.  **Rating 1 (Fail):** Contradicts the `context` (e.g., says 'pain' when context says 'no pain').
    3.  **Rating 1 (Fail):** Asks a question back to the doctor.
    4.  **Rating 2-3 (Poor):** Evasive, robotic, or irrelevant, but otherwise follows rules.
    5.  **Rating 4-5 (Good):** Direct, natural, consistent, and correct answer.
    """
    question: str = dspy.InputField(description="Doctor’s question (Hindi).")
    response: str = dspy.InputField(description="Patient’s response (Hindi).")
    context: str = dspy.InputField(description="Patient’s medical history.")
    retrieved_history: Optional[str] = dspy.InputField(description="Relevant conversation snippets.")
    response_rating: int = dspy.OutputField(
        description="A single integer rating 1 (FAIL) to 5 (Excellent).",
        le=5, ge=1
    )


# ========== Critique Signature ==========
class Critique(dspy.Signature):
    """
    Role: Act as a strict error-checker. Provide a *single* actionable command (in English).

    **Error Priorities:**
    1.  **Language Purity:** Contains Roman/English/Jargon (e.g., `मूकोपुरulent`, `problem`).
        *Feedback:* "Response contains Hinglish or jargon. Re-write using only simple, pure
        Hindi descriptions from the context (e.g., use 'मलाई जैसा' instead of 'mucopurulent')."
    2.  **Consistency:** Contradicts the `context`.
        *Feedback:* "Response contradicts medical history. Re-write to be consistent with context."
    3.  **Persona:** Asks a question.
        *Feedback:* "Do not ask questions. Re-phrase as a direct answer."
    4.  **Naturalness:** Robotic, evasive, or nonsensical.
        *Feedback:* "Response is unnatural or evasive. Provide a direct, natural answer."

    **Pass Condition:**
    -   If the response is perfect (pure Hindi, consistent, natural),
        you **MUST** output the exact string: `No improvement needed.`
    """
    question: str = dspy.InputField(description="Doctor's question.")
    response: str = dspy.InputField(description="Patient's Hindi response.")
    context: str = dspy.InputField(description="Patient’s medical history.")
    retrieved_history: Optional[str] = dspy.InputField(description="Relevant conversation snippets.")
    feedback: str = dspy.OutputField(
        description="A single feedback command (in English) or 'No improvement needed.'"
    )


# ========== Improvement Signature ==========
class Improvement(QAPatient):
    """
    Role: Revise the `previous_response` based *only* on the `feedback`.

    **CRITICAL Rules:**
    1.  **Pass Condition:** If `feedback` is exactly `No improvement needed.`,
        you **MUST** output the `previous_response` with zero changes.
    2.  **Apply Feedback:** Apply the specific feedback (e.g., remove jargon, fix contradiction).
    3.  **Obey Persona:** The new `answer` **MUST** still obey all `QAPatient` rules.
    """
    feedback: str = dspy.InputField(description="The actionable feedback command (in English).")
    previous_response: str = dspy.InputField(description="Original patient response.")
    # Inherits question, context, retrieved_history, and answer


# ========== FailCase Signature ==========
class FailCase(dspy.Signature):
    """
    Role: Provide a polite, natural, fallback response (in Hindi) if the patient is confused.

    Example: "माफ़ कीजिये, मुझे समझ नहीं आया कि आप क्या पूछ रहे हैं।"
    """
    fail_response: str = dspy.OutputField(description="Fallback Hindi response (Devanagari).")


class RAGmodule(dspy.Module):
    def __init__(self):
        self.response = dspy.Predict(QAPatient)
        self.retrieve_history = dspy.ChainOfThought(HistoryRetriever)
        self.response_rating = dspy.ChainOfThought(ResponseJudge)
        self.response_failure = dspy.Predict(FailCase)
        self.responce_feedback = dspy.ChainOfThought(Critique)
        self.imporve_responce = dspy.Predict(Improvement)
    def forward(self,query:str):
        context = docs
        retrieved_history = self.retrieve_history(question=query,context=context,history=history)
        retrieved_history = retrieved_history.retrieved_history
        response = self.response(question=query,context=context,retrieved_history=retrieved_history)
        response = response.answer
        for _ in range(8):
            responce_feedback = self.responce_feedback(question=query,response=response,context=context,retrieved_history=retrieved_history)
            response = self.imporve_responce(question=query,context=context,feedback=responce_feedback.feedback,retrieved_history=retrieved_history,previous_response=response) 
            response = response.answer
        responce_rating = self.response_rating(question=query,response=response,context=context,retrieved_history=retrieved_history)
        print(f"Response rating:\n{responce_rating}")
        if responce_rating.response_rating<4:
            fail_response = self.response_failure().fail_response
            return fail_response
        history.messages.append({"question": query, "Patient":response})
        return response
        


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
        
    