# AI Agent for Automated HR Policy QA and Grievances

This implementation follows the supplied project synopsis: policy-document chunking, vector retrieval, source-aware answers, grievance classification, priority/severity prediction, department prediction, and status tracking.

## 1. Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Run

From the project root:

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

The first run downloads the `all-MiniLM-L6-v2` embedding model from Hugging Face. After that, the model is cached locally.

## 3. Features

- HR policy Q&A using FAISS vector retrieval.
- Source document and section metadata in the answer.
- Grievance category classification.
- Grievance severity classification.
- Department prediction from the supplied grievance data.
- SQLite grievance storage.
- Open / In Progress / Resolved workflow.
- Simple browser dashboard.

## 4. API

- `POST /api/policy/ask`
- `POST /api/grievances/predict`
- `POST /api/grievances`
- `GET /api/grievances`
- `PATCH /api/grievances/{grievance_id}/status`
- Interactive API docs: `/docs`

## 5. Important academic limitation

The supplied synopsis describes an LLM-generated grounded answer as the proposed final stage. This starter implementation performs retrieval and returns the strongest retrieved policy passage rather than calling a paid external LLM. An LLM can be added later behind `PolicyRAG.answer()` if required.
