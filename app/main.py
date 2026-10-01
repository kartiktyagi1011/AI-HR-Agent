from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.rag import PolicyRAG
from app.grievance import GrievanceModel
from app.database import init_db, create_grievance, list_grievances, update_status

app = FastAPI(title="AI Agent for HR Policy QA and Grievances", version="1.0.0")
templates = Jinja2Templates(directory="templates")

rag = PolicyRAG()
grievance_model = GrievanceModel()
init_db()

class Question(BaseModel):
    question: str = Field(min_length=3)

class GrievanceRequest(BaseModel):
    employee_name: str = "Anonymous"
    description: str = Field(min_length=5)

class StatusRequest(BaseModel):
    status: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/policy/ask")
def ask_policy(payload: Question):
    return rag.answer(payload.question)

@app.post("/api/grievances/predict")
def predict_grievance(payload: GrievanceRequest):
    return grievance_model.predict(payload.description)

@app.post("/api/grievances")
def submit_grievance(payload: GrievanceRequest):
    prediction = grievance_model.predict(payload.description)
    grievance_id = "GRV-" + uuid4().hex[:8].upper()
    data = {
        "grievance_id": grievance_id,
        "employee_name": payload.employee_name,
        "description": payload.description,
        **prediction,
        "status": "Open",
    }
    create_grievance(data)
    return data

@app.get("/api/grievances")
def get_grievances():
    return list_grievances()

@app.patch("/api/grievances/{grievance_id}/status")
def change_status(grievance_id: str, payload: StatusRequest):
    allowed = {"Open", "In Progress", "Resolved"}
    if payload.status not in allowed:
        raise HTTPException(400, "Status must be Open, In Progress, or Resolved")
    if not update_status(grievance_id, payload.status):
        raise HTTPException(404, "Grievance not found")
    return {"message": "Status updated", "grievance_id": grievance_id, "status": payload.status}
