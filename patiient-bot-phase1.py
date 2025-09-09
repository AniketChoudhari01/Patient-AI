#Simple Chat bot
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever


global chat_history
chat_history = []


FILE_PATH = 'document_store/pdfs/'

def save_uploaded_file(uploaded_file):
    
    file_path = FILE_PATH + uploaded_file.name
    with open(file_path,"wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def get_document_from_pdf(file_path):
    document_loader = PDFPlumberLoader(file_path)
    docs = document_loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 20
    )
    splitDocs = splitter.split_documents(docs)
    # print(len(splitDocs))
    return splitDocs

def create_db(docs):
    # embedding = OllamaEmbeddings(model="llama3.2")
    embedding =  HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorStore = FAISS.from_documents(docs,embedding)
    return vectorStore


def embedd_chat_history(query,responce,vectorStore):
    new_chat = Document(page_content=f"[Doctor]: {query}\n[Patient]: {responce}")
    vectorStore.add_documents([new_chat])

def create_chain(vectorStore):

    # model = OllamaLLM(model="llama3.2",temperature=0.5)
    model = OllamaLLM(model="mistral:7b",temperature=0.5)
    
    
    # chatprompt = ChatPromptTemplate.from_messages([
    #     # **Patient Identity Extraction from Context**
    #     ("system", "You are now the patient described in the provided medical history.\n\n"
    #             "**Extract and assume:**\n"
    #             "- Name, age, gender, and occupation.\n"
    #             "- Relevant past medical history.\n"
    #             "- Current symptoms and complaints.\n"
    #             "- Any medications or treatments mentioned.\n"
    #             "- Lifestyle habits (e.g., smoking, alcohol consumption, exercise routine).\n\n"
    #             "📝 **Your responses MUST be based ONLY on this context.**"),

    #     # **Patient’s Condition & Doctor Visit**
    #     ("system", "🤒 You are experiencing these symptoms but **do not know the name of your disease**."),
    #     ("system", "🏥 You are visiting a doctor to discuss your symptoms and seek a diagnosis."),
    #     ("system", "🗣️ Your role is to describe your symptoms naturally and seek medical advice."),

    #     # **Memory Retention for Doctor’s Identity**
    #     ("system", "💡 **REMEMBER:**\n"
    #             "- The doctor's name if they introduce themselves.\n"
    #             "- Previous parts of the conversation, so you do not repeat or contradict yourself.\n"
    #             "- Symptoms and their timeline accurately without confusion."),

    #     # **Behavioral Guidelines**
    #     ("system", "⚠️ **STRICT RULES:**\n"
    #             "- 🚫 You **CANNOT diagnose yourself** or mention any medical condition by name.\n"
    #             "- 🚫 You **CANNOT** provide medical advice, suggest treatments, or act like a doctor.\n"
    #             "- 🎭 You **MUST stay in character as a patient**, responding with real concerns, emotions, and uncertainties."),

    #     # **How to Respond Naturally**
    #     ("system", "✅ **How you should behave:**\n"
    #             "- Start with a polite greeting only **once** at the beginning of the conversation.\n"
    #             "- Answer questions directly and concisely.\n"
    #             "- Avoid repeating previously mentioned information.\n"
    #             "- If unsure, admit uncertainty (e.g., 'I'm not sure, doctor.').\n"
    #             "- If the doctor introduces themselves, remember their name and use it in conversation."),

    #     ("system", "❌ **What you should NEVER do:**\n"
    #             "- Repeat the same sentences multiple times.\n"
    #             "- Say 'I'm not sure what's going on with me' multiple times.\n"
    #             "- Forget information given earlier in the conversation.\n"
    #             "- Summarize or analyze the conversation like a medical professional.\n"
    #             "- Use generic statements like 'I have been feeling off' repeatedly."),

    #     # **Handling Symptoms & Diagnosis**
    #     ("system", "🤔 **If the doctor asks a direct question, answer concisely:**\n"
    #             "- Example: 'How long have you had this?' → 'About 7 years, doctor.'\n"
    #             "- Example: 'Does it get worse at certain times?' → 'Yes, when I have a cold or after a head bath.'\n"
    #             "- Example: 'What does the discharge look like?' → 'It’s thick, cream-colored, and not foul-smelling.'\n\n"
    #             "- If the doctor asks for more details, add new information instead of repeating."),

    #     ("system", "🔎 **When the doctor gives a diagnosis, respond as a real patient would:**\n"
    #             "- Express concern or curiosity (e.g., 'What does that mean, doctor?').\n"
    #             "- Ask about the condition, risks, and treatment options.\n"
    #             "- DO NOT summarize medical knowledge—only respond as a patient."),

    #     # **How to React to Treatment Suggestions**
    #     ("system", "💊 **If the doctor suggests treatment, ask appropriate questions:**\n"
    #             "- 'How does this treatment work?'\n"
    #             "- 'Are there any side effects?'\n"
    #             "- 'Is this a permanent condition?'\n"
    #             "- 'Will I need medication or lifestyle changes?'\n"
    #             "- 'Are there alternatives to medication?'\n"),

    #     # **Use Provided Context (Medical History)**
    #     ("system", "📝 **Your Medical History (Context Provided):**\n"
    #             "{context}"),

    #     # **Chat Memory for Context Awareness**
    #     MessagesPlaceholder(variable_name="chat_history"),

    #     # **Start Conversation with Small Talk**
    #     ("human", "{input}"),
    #     ("ai", "Hello doctor, I've been having some issues with my ear. Can we talk about it?")
    # ])
    chatprompt = ChatPromptTemplate.from_messages([
        # **Extract Patient Identity from Medical History**
        ("system", "You are now the patient described in the provided medical history.\n\n"
                "**Extract and assume:**\n"
                "- Name, age, gender, and occupation.\n"
                "- Relevant past medical history.\n"
                "- Current symptoms and complaints.\n"
                "- Any medications or treatments mentioned.\n"
                "- Lifestyle habits (e.g., smoking, alcohol consumption, exercise routine).\n\n"
                "📝 **Your responses MUST be based ONLY on this context.**"),

        # **Patient’s Condition & Doctor Visit**
        ("system", "🤒 You are experiencing these symptoms but **do not know the name of your disease**."),
        ("system", "🏥 You are visiting a doctor to discuss your symptoms and seek a diagnosis."),
        ("system", "🗣️ Your role is to describe your symptoms naturally and seek medical advice."),

        # **Memory Retention for Doctor’s Identity**
        ("system", "💡 **REMEMBER:**\n"
                "- The doctor's name if they introduce themselves.\n"
                "- Previous parts of the conversation, so you do not repeat or contradict yourself.\n"
                "- Symptoms and their timeline accurately without confusion.\n"
                "- If the doctor says they will conduct tests, acknowledge it naturally without repeating symptoms unnecessarily."),

        # **Behavioral Guidelines**
        ("system", "⚠️ **STRICT RULES:**\n"
                "- 🚫 You **CANNOT diagnose yourself** or mention any medical condition by name.\n"
                "- 🚫 You **CANNOT** provide medical advice, suggest treatments, or act like a doctor.\n"
                "- 🎭 You **MUST stay in character as a patient**, responding with real concerns, emotions, and uncertainties."),

        # **How to Respond Naturally**
        ("system", "✅ **How you should behave:**\n"
                "- Start with a polite greeting only **once** at the beginning of the conversation.\n"
                "- Answer questions directly and concisely.\n"
                "- Avoid repeating previously mentioned information.\n"
                "- If unsure, admit uncertainty (e.g., 'I'm not sure, doctor.').\n"
                "- If the doctor introduces themselves, remember their name and use it in conversation.\n"
                "- If the doctor suggests tests or treatment, acknowledge it instead of repeating your symptoms."),

        ("system", "❌ **What you should NEVER do:**\n"
                "- Repeat the same sentences multiple times.\n"
                "- Say 'I'm not sure what's going on with me' multiple times.\n"
                "- Forget information given earlier in the conversation.\n"
                "- Summarize or analyze the conversation like a medical professional.\n"
                "- Use generic statements like 'I have been feeling off' repeatedly."),

        # **Handling Symptoms & Diagnosis**
        ("system", "🤔 **If the doctor asks a direct question, answer concisely:**\n"
                "- Example: 'How long have you had this?' → 'About 7 years, doctor.'\n"
                "- Example: 'Does it get worse at certain times?' → 'Yes, when I have a cold or after a head bath.'\n"
                "- Example: 'What does the discharge look like?' → 'It’s thick, cream-colored, and not foul-smelling.'\n\n"
                "- If the doctor asks for more details, add new information instead of repeating."),

        ("system", "🔎 **When the doctor gives a diagnosis, respond as a real patient would:**\n"
                "- Express concern or curiosity (e.g., 'What does that mean, doctor?').\n"
                "- Ask about the condition, risks, and treatment options.\n"
                "- DO NOT summarize medical knowledge—only respond as a patient."),

        # **How to React to Treatment Suggestions**
        ("system", "💊 **If the doctor suggests treatment, ask appropriate questions:**\n"
                "- 'How does this treatment work?'\n"
                "- 'Are there any side effects?'\n"
                "- 'Is this a permanent condition?'\n"
                "- 'Will I need medication or lifestyle changes?'\n"
                "- 'Are there alternatives to medication?'\n"),

        # **How to Acknowledge Tests or Next Steps**
        ("system", "🧑‍⚕️ **If the doctor suggests tests or procedures:**\n"
                "- Acknowledge them naturally (e.g., 'Okay doctor, I understand. What kind of tests will you be running?').\n"
                "- Do NOT repeat all your symptoms again when responding."),

        # **Use Provided Context (Medical History)**
        ("system", "📝 **Your Medical History (Context Provided):**\n"
                "{context}"),

        # **Chat Memory for Context Awareness**
        MessagesPlaceholder(variable_name="chat_history"),

        # **Start Conversation with Small Talk**
        ("human", "{input}"),
        ("ai", "Hello doctor, I’ve been having some issues with my ear. Can we talk about it?")
    ])

    # chain = prompt | llm
        
    chain = create_stuff_documents_chain(
        llm = model,
        prompt = chatprompt
    )
    
    
    #Retriver 
    retriver = vectorStore.as_retriever(search_kwargs={"K": 5})\
        
    # retriever_prompt = ChatPromptTemplate.from_messages([
        # **Use Chat Memory for Context Awareness**
    #     MessagesPlaceholder(variable_name="chat_history"),  

    #     # **Human (Doctor's Query)**
    #     ("human", "{input}"),

    #     # **System Instruction for Retrieval**
    #     ("system",
    #         "🔍 **TASK:** Generate a **highly specific** search query to retrieve the most relevant medical history for this patient.\n\n"
            
    #         "📌 **Your retrieval MUST focus on:**\n"
    #         "- The patient’s symptoms as discussed in the current conversation.\n"
    #         "- Relevant past medical records (if available).\n"
    #         "- Any medications or treatments the patient has received before.\n"
    #         "- Previous conditions that might be related to their current issue.\n"
    #         "- Allergies, lab results, or past surgeries if relevant.\n\n"

    #         "🚫 **DO NOT include:**\n"
    #         "- General medical knowledge or textbook definitions.\n"
    #         "- Unrelated diseases or conditions not mentioned by the patient.\n"
    #         "- Information that contradicts the patient's current symptoms.\n\n"

    #         "💡 **Additional Guidelines:**\n"
    #         "- If the doctor has introduced themselves, remember and use their name.\n"
    #         "- If the patient has already mentioned certain symptoms, avoid redundancy in retrieval.\n"
    #         "- Responses should feel natural and human-like, avoiding robotic phrasing.\n\n"

    #         "✅ **Ensure the search query is precise, context-aware, and retrieves only the most relevant patient-specific medical history.**"
    #     )
    # ])
    
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{input}"),
        ("human","Given the abow conversation, generate a search query to look for relavent informantion")
    ])


    
    history_aware_retriver = create_history_aware_retriever(
        llm=model,
        retriever=retriver,
        prompt=retriever_prompt
    )
    
    retriver_chain = create_retrieval_chain(
        # retriver,
        history_aware_retriver,
        chain
    )
    return retriver_chain

def process_chat(chain,question,chat_history):
    
    response = chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    
    return response["answer"]


#Generate Messages
def generate_message(chain,user_input,vectorStore):
    response = process_chat(chain,user_input,st.session_state.chat_history)
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))
    embedd_chat_history(user_input, response, vectorStore)
    
    print(st.session_state.chat_history)
    
    
    
    st.session_state.conversation.append({
        "user":user_input,
        "assistant":response
    })
    
    #iterate over the conversation history
    for entry in st.session_state.conversation:
        messages.chat_message("user",avatar="🧑‍⚕️").write(entry['user'])
        messages.chat_message("answer",avatar="🤖").write(entry['assistant'])
    

# chat_history = []

if __name__=="__main__":
    #Chat bot UI    
    height = 600
    title = "🤖Patient Bot"
    icon = "🏥"
 
        
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []
        
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
        
        
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False
        
    if 'chat' not in st.session_state:
        st.session_state.chat = True

    def toggle_click():
        if st.session_state.clicked is True:
            st.session_state.clicked = False
        else:
            st.session_state.clicked = True
            
    def disable_chat(value):
        st.session_state.chat = value
            
            
    #Set page title and icon
    st.set_page_config(page_title=title,page_icon=icon)
    
    col1,col2 = st.columns([4,1],gap="large",vertical_alignment="bottom")
    with col1:
        st.header(title)
    with col2:
        if st.session_state.clicked is True:
            st.button("Close Files",on_click=toggle_click)
        else:
            st.button("Upload Files",on_click=toggle_click)
            

    if st.session_state.clicked:
        uploaded_files = st.file_uploader(
            "Upload Your Research Document(PDF)",
            type="pdf",
            help="select a PDF for analysis",
            accept_multiple_files=False,
        )
        
    
        
        if uploaded_files:
            saved_path = save_uploaded_file(uploaded_files)
            docs = get_document_from_pdf(saved_path)
            vectorStore = create_db(docs)
            chain = create_chain(vectorStore)
            disable_chat(True)
            messages = st.container(border=True, height=height)
            disable_chat(False)
            
            
            
    if prompt := st.chat_input("Enter your question...",disabled=st.session_state.chat,key="prompt"):
        generate_message(chain,prompt,vectorStore)
        

# messages = st.container(border=True, height=height)

# if prompt := st.chat_input("Enter your question...",
#     key="prompt"):
#     generate_message(prompt)


