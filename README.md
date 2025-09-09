# 🤖 Patient AI – Virtual Medical Training Assistant  

## Table of Contents  
- [Overview](#overview)  
- [Project Goals](#project-goals)  
- [Features](#features)  
- [Tech Stack](#tech-stack)  
- [Current Workflow](#current-workflow)  
- [Screenshots](#screenshots)  
  - [English Prototype](#english-prototype-achieved--step-1)  
  - [Hindi Prototype](#hindi-prototype-work-in-progress--step-2)  
- [Installation & Usage](#installation--usage)  
- [Future Vision](#future-vision)  
- [Why This Project Matters](#why-this-project-matters)  
- [Team Members](#team-members)  

---

## Overview  
Patient AI is a **virtual patient chatbot** designed to help **medical students practice patient-doctor conversations** in a safe, controlled, and private environment.  

The project allows students to upload **past patient histories (PDFs)**, which are stored in a **local vector database**. Students can then interact with the AI in a **realistic patient roleplay**, asking questions and receiving responses that are **context-aware** and based on the uploaded documents.  

🚫 **No data breaches** – The system runs entirely **offline and locally**, with **no external API calls**.  

---

## Project Goals  
- ✅ **Step 1:** Develop an English-based chatbot (achieved).  
- 🚧 **Step 2:** Extend support to **regional languages** (next target: Hindi).  
   - Students will type in **Hinglish** (e.g., "aapko kab se bukhar hai?").  
   - AI will respond in **strict Devanagari Hindi**.  
- 🔮 **Future Roadmap:**  
   - Build a **Streamlit UI** for a smooth end-to-end app experience.  
   - Implement **student login and session tracking**.  
   - Add an **LLM Judge** that evaluates and scores medical students based on their questioning skills.  
   - Expand to multiple regional languages.  

---

## Features  
- 📂 **PDF Upload** – Import patient histories as training context.  
- 🧠 **Vector Database (FAISS)** – Efficient storage & retrieval of medical history.  
- 🗣️ **Conversational AI** – Acts as the patient in a roleplay scenario.  
- 🌐 **Language Expansion** – English now, Hindi and others in the pipeline.  
- 🛡️ **Local Setup** – No internet or external API calls required.  

---

## Tech Stack  
- [Python](https://www.python.org/)  
- [Streamlit](https://streamlit.io/) – Interactive UI  
- [LangChain](https://www.langchain.com/) – Orchestration & retrieval chains  
- [Ollama](https://ollama.ai/) – Local LLM & embeddings  
- [FAISS](https://faiss.ai/) – Vector store for document embeddings  
- [PDFPlumber](https://github.com/jsvine/pdfplumber) – PDF document loader  

---

## Current Workflow  
1. Medical student uploads a **PDF patient history**.  
2. Document is **split into chunks** and stored in a **vector database (FAISS)**.  
3. Student starts a **chat session**.  
4. AI **acts as the patient** and answers only based on the given context.  
5. Chat history is also stored in the vector DB for **better continuity**.  

---

## Screenshots  

### English Prototype (Achieved – Step 1)  
<img width="1169" height="791" alt="image" src="https://github.com/user-attachments/assets/14225306-a89c-4199-a23b-ab0de348cfd0" />


### Hindi Prototype (Work in Progress – Step 2)  
*(Add placeholder for future Hindi screenshots)*  

---

## Installation & Usage  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/AniketChoudhari01/patient-ai.git
