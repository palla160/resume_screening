from fastapi import FastAPI, UploadFile, File, Form
import pdfplumber
import requests
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# ---------------- CONFIG ----------------

app = FastAPI(title="HR Resume Screening API")

# ---------------- LLM ----------------
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ---------------- LOAD SKILLS ----------------
with open("skills.txt") as f:
    REQUIRED_SKILLS = [s.strip().lower() for s in f.readlines()]

# ---------------- STATE ----------------
class ResumeState(TypedDict):
    resume_text: str
    extracted_skills: Optional[List[str]]
    score: Optional[int]
    explanation: Optional[str]
    decision: Optional[str]

# ---------------- PDF READER ----------------
def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.lower()

# ---------------- AGENTS ----------------
def extract_agent(state: ResumeState):

    prompt = f"""
Extract technical skills from this resume:

{state['resume_text']}

Return comma-separated skills.
"""

    result = llm.invoke(prompt).content.lower()
    skills = [s.strip() for s in result.split(",")]

    return {"extracted_skills": skills}

def score_agent(state: ResumeState):

    match = len(set(state["extracted_skills"]) & set(REQUIRED_SKILLS))
    total = len(REQUIRED_SKILLS)
    score = int((match / total) * 100) if total else 0

    return {"score": score}

def explain_agent(state: ResumeState):

    prompt = f"""
Candidate skills: {state['extracted_skills']}
Required skills: {REQUIRED_SKILLS}
Score: {state['score']}

based on the above skills create one summary which is suitable for sending the mail to the candidate.
it will look like a real mail.where real hrs send message to the candidates when they selected for the job.

notr:
* Dont add name, email,comany name,starting date,salary.
*just create one summary that you are elgibe or not fpor this role based on Score: {state['score']}.
*dont mention best regards also.
"""

    explanation = llm.invoke(prompt).content
    return {"explanation": explanation}

def decision_agent(state: ResumeState):

    decision = "Selected" if state["score"] >= 60 else "Rejected"
    return {"decision": decision}

# ---------------- GRAPH ----------------
graph = StateGraph(ResumeState)

graph.add_node("extract", extract_agent)
graph.add_node("score", score_agent)
graph.add_node("explain", explain_agent)
graph.add_node("decision", decision_agent)

graph.add_edge(START, "extract")
graph.add_edge("extract", "score")
graph.add_edge("score", "explain")
graph.add_edge("explain", "decision")
graph.add_edge("decision", END)

resume_graph = graph.compile()

# ---------------- API ----------------
@app.post("/apply")
async def apply(
    name: str = Form(...),
    email: str = Form(...),
    resume: UploadFile = File(...)
):

    # 1. Extract resume text
    text = read_pdf(resume.file)

    # 2. Run LangGraph ONLY on resume
    result = resume_graph.invoke({
        "resume_text": text
    })

    # 3. Merge with applicant info
    payload = {
        "name": name,
        "email": email,
        "score": result["score"],
        "decision": result["decision"],
        "explanation": result["explanation"]
    }

    return payload