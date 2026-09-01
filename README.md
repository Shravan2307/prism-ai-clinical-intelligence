# 🏥 Medical AI Clinical Intelligence

An AI-powered clinical intelligence platform designed to assist with **early disease-risk identification, clinical analysis, treatment recommendations, and intelligent patient triage**.

The system combines an existing Machine Learning/AI implementation with a structured application layer built using **FastAPI**, allowing patient information to flow through safety checks, triage, clinical analysis, and the existing ML components.

> ⚠️ **Medical Disclaimer:** This project is a clinical decision-support prototype intended for research, education, and hackathon purposes. It is not a replacement for a qualified medical professional, diagnosis, or emergency medical care.

---

## 🚀 Project Overview

The goal of the platform is to provide an intelligent workflow that can:

1. Accept structured patient information.
2. Perform an initial safety assessment.
3. Determine the appropriate triage workflow.
4. Run the existing clinical ML/AI components.
5. Generate structured clinical insights.
6. Provide risk and contributing-factor information.
7. Provide treatment/recommendation information where supported.
8. Escalate appropriate cases to a doctor workflow.
9. Present the results through a modern frontend dashboard.

### Core Principle

The application layer **does not duplicate the existing ML logic**.

The existing ML implementation remains the source of truth for:

* Disease/risk prediction
* Risk scoring
* Treatment ranking
* Causal analysis
* Clinical intelligence

The application layer is responsible for orchestrating these components and exposing them through a clean API.

---

# 🧠 System Architecture

```text
                    ┌─────────────────────┐
                    │      Frontend       │
                    │  Clinical Dashboard │
                    └──────────┬──────────┘
                               │
                               │ HTTP / JSON
                               ▼
                    ┌─────────────────────┐
                    │     FastAPI API     │
                    │   Application Layer │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   PatientRequest    │
                    │   Pydantic Model    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Safety Gate     │
                    │ Emergency/Urgency   │
                    │     Detection       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Triage Engine    │
                    │ Workflow Selection  │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │  ClinicalAnalysisService  │
                 │       Orchestrator        │
                 └─────────────┬─────────────┘
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
      ┌────────────┐    ┌────────────┐    ┌────────────┐
      │ Risk/ML    │    │ Treatment  │    │  Causal    │
      │ Engine     │    │ Ranking    │    │  Analysis  │
      └────────────┘    └────────────┘    └────────────┘
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Structured Clinical │
                    │      Response       │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
       ┌────────────────┐             ┌────────────────┐
       │ Patient Result │             │ Doctor Workflow│
       │   Dashboard    │             │   / Escalation │
       └────────────────┘             └────────────────┘
```

---

# ✨ Key Features

## 1. Patient Analysis

The frontend collects the information required by the backend and sends it to the FastAPI API.

The application uses a structured Pydantic request model to validate incoming patient information.

---

## 2. Safety Gate

Before normal clinical processing, the application performs a safety check.

The Safety Gate is responsible for identifying situations that may require:

* Emergency escalation
* Urgent medical review
* Doctor review
* Normal clinical workflow

The frontend does **not** independently calculate safety or risk.

The backend Safety Gate is the source of truth.

---

## 3. Intelligent Triage

The Triage Engine determines the next workflow based on the backend analysis.

Possible workflow categories can include:

```text
Routine
Monitor
Follow-up
Doctor Review
Urgent
Emergency
```

The exact workflow is determined by the backend implementation.

---

## 4. Clinical Analysis Service

`ClinicalAnalysisService` acts as the main orchestration layer.

Its responsibility is to connect:

```text
Patient Request
      ↓
Safety Gate
      ↓
Triage Engine
      ↓
Existing ML/Epics
      ↓
Clinical Result
```

This keeps the application layer separate from the underlying ML implementation.

---

## 5. Existing ML Components

The existing ML implementation is preserved.

The project does **not** create duplicate implementations of:

* Risk Engine
* Treatment Ranking Engine
* Causal Engine
* Disease Prediction
* Clinical Reasoning

Instead, the application layer adapts to the existing ML interfaces.

---

## 6. Doctor Escalation Workflow

When the backend identifies a case requiring professional review, the system can route the case toward the doctor workflow.

The doctor dashboard can display:

* Patient information
* Triage status
* Safety status
* Clinical findings
* Risk information
* Contributing factors
* Recommendations
* Relevant analysis

The frontend displays information supplied by the backend rather than creating independent medical decisions.

---

# 🏗️ Project Structure

The project is organized into separate backend and frontend layers.

```text
prism-ai-clinical-intelligence/
│
├── backend/
│   │
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/
│   │   │
│   │   ├── models/
│   │   │
│   │   ├── schemas/
│   │   │
│   │   ├── services/
│   │   │
│   │   ├── decision/
│   │   │
│   │   ├── agents/
│   │   │
│   │   ├── core/
│   │   │
│   │   └── main.py
│   │
│   ├── tests/
│   │
│   └── requirements.txt
│
├── frontend/
│   └── ...
│
├── .gitignore
├── README.md
└── ...
```

> The exact directories may evolve as the application layer and frontend integration are completed.

---

# ⚙️ Technology Stack

## Backend

* Python
* FastAPI
* Pydantic
* Uvicorn
* Pytest
* Existing ML/AI components

## Frontend

* JavaScript/TypeScript
* React-based frontend
* Modern responsive UI
* REST API integration

## Architecture

* REST API
* Service-oriented application layer
* Pydantic request/response validation
* ML/AI orchestration
* Safety-first clinical workflow

---

# 📋 Prerequisites

Install the following before running the project.

### Python

Recommended:

```text
Python 3.11+
```

The project has also been developed/tested in the current environment using Python 3.14.x.

Check your version:

```cmd
python --version
```

### Node.js

Check:

```cmd
node --version
npm --version
```

---

# 🔧 Backend Setup

Open a terminal in the project root:

```cmd
cd prism-ai-clinical-intelligence
```

Move into the backend:

```cmd
cd backend
```

---

## 1. Create Virtual Environment

Windows:

```cmd
python -m venv venv
```

Activate it:

```cmd
venv\Scripts\activate
```

You should see:

```text
(venv)
```

at the beginning of your terminal prompt.

---

## 2. Install Dependencies

```cmd
pip install -r requirements.txt
```

If the project does not contain a generated requirements file yet, install the project's declared dependencies according to the backend configuration.

---

## 3. Environment Variables

Create a `.env` file if required by the project.

Example:

```env
# Backend configuration
APP_ENV=development

# Frontend URL
FRONTEND_URL=http://localhost:5173

# Add project-specific API keys/configuration here
# Never commit real secrets to GitHub.
```

Never commit:

```text
.env
```

to GitHub.

Use `.env.example` for non-secret configuration documentation.

---

# ▶️ Running the Backend

From the `backend` directory:

```cmd
python -m uvicorn app.main:app --reload
```

The backend should start at:

```text
http://127.0.0.1:8000
```

FastAPI interactive documentation:

```text
http://127.0.0.1:8000/docs
```

Alternative ReDoc documentation:

```text
http://127.0.0.1:8000/redoc
```

---

# 🎨 Frontend Setup

Open a **second terminal**.

Move to the frontend directory containing `package.json`.

For example:

```cmd
cd prism-ai-clinical-intelligence\frontend
```

First verify:

```cmd
dir package.json
```

If `package.json` exists, install dependencies:

```cmd
npm install
```

Then start the frontend:

```cmd
npm run dev
```

The development server will normally be available at:

```text
http://localhost:5173
```

> If `frontend\package.json` does not exist, locate the actual frontend directory first:
>
> ```cmd
> cd C:\Users\Shravan\prism-ai-clinical-intelligence
> dir /s /b package.json
> ```
>
> Run `npm install` from the directory containing the frontend's `package.json`.

---

# 🔗 Frontend ↔ Backend Configuration

The frontend should communicate with FastAPI through an environment variable.

For a Vite frontend, use:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

The frontend API client should use this variable instead of hardcoding the backend URL throughout the application.

Example:

```text
Frontend
   │
   │ POST /<clinical-analysis-endpoint>
   ▼
FastAPI
   │
   ▼
ClinicalAnalysisService
   │
   ▼
Existing ML Implementation
```

The exact endpoint should match the backend router implemented in the project.

---

# 🔄 Complete Application Flow

## Normal Case

```text
Patient enters information
          ↓
Frontend validation
          ↓
POST request to FastAPI
          ↓
PatientRequest validation
          ↓
Safety Gate
          ↓
Triage Engine
          ↓
ClinicalAnalysisService
          ↓
Existing ML components
          ↓
Clinical response
          ↓
Frontend results dashboard
```

---

## Urgent/Emergency Case

```text
Patient enters information
          ↓
FastAPI
          ↓
Safety Gate
          ↓
Urgent/Emergency detected
          ↓
Triage decision
          ↓
Escalation workflow
          ↓
Doctor review
```

The frontend should clearly communicate the escalation status returned by the backend.

---

# 🧪 Testing

Run the backend tests from the `backend` directory:

```cmd
python -m pytest
```

For verbose output:

```cmd
python -m pytest -v
```

The project should maintain existing tests while adding tests for the application layer.

Important test areas include:

* Patient request validation
* Safety Gate
* Triage
* ClinicalAnalysisService
* API routes
* Successful clinical analysis
* Error handling
* Doctor escalation workflow
* Existing ML functionality

---

# 🔌 API

The API is built using FastAPI.

After starting the backend, open:

```text
http://127.0.0.1:8000/docs
```

This provides an interactive Swagger interface where available API endpoints can be tested.

The primary analysis flow follows the conceptual structure:

```http
POST /<clinical-analysis-endpoint>
```

Request:

```json
{
  "patient_information": "..."
}
```

The exact request fields must match the project's `PatientRequest` Pydantic model.

The response contains structured clinical information generated by the backend.

---

# 🛡️ Safety & Clinical Design Principles

This project follows a **safety-first architecture**.

### Backend is authoritative

Clinical decisions are generated by backend components.

The frontend must never override backend safety decisions.

### No duplicate medical logic

There must be only one authoritative implementation of each clinical reasoning component.

### Explicit escalation

Potentially urgent cases are routed through the Safety Gate and Triage Engine before normal workflows continue.

### Human-in-the-loop

Doctor review is used when the system determines that professional evaluation is appropriate.

### No fabricated results

The application must never display fake predictions, fake patient data, or fabricated clinical recommendations as real results.

---

# 🔐 Security

Do not commit secrets to GitHub.

The following should remain local:

```text
.env
API keys
private credentials
database credentials
```

Recommended `.gitignore` entries:

```gitignore
.env
.env.*
!.env.example

__pycache__/
*.py[cod]

.venv/
venv/

node_modules/

*.log

.DS_Store
```

---

# 🧩 Development Principles

When extending the project:

### DO

* Reuse existing ML components.
* Reuse existing interfaces.
* Keep API contracts stable.
* Add Pydantic validation.
* Add tests.
* Keep frontend/backend responsibilities separate.
* Use environment variables.
* Preserve existing functionality.

### DO NOT

* Create a second risk engine.
* Create a second treatment ranking engine.
* Create a second causal engine.
* Move ML logic into React.
* Hardcode clinical predictions.
* Fake API responses in production.
* Commit secrets.
* Break existing tests.
* Rewrite working ML components unnecessarily.

---

# 🛠️ Development Roadmap

The application is being developed in the following phases:

```text
PHASE A — INSPECT
       ↓
PHASE B — APPLICATION MODELS
       ↓
PHASE C — SAFETY
       ↓
PHASE D — SERVICE LAYER
       ↓
PHASE E — API
       ↓
PHASE F — DOCTOR WORKFLOW
       ↓
PHASE G — REPORT
       ↓
PHASE H — TESTS
       ↓
PHASE I — CLEANUP
```

---

# 📌 Current Integration Strategy

The project uses a **layered architecture**.

The existing ML implementation is preserved as the core intelligence layer.

The new application layer adapts the existing interfaces into a clean workflow:

```text
Existing ML
     ↑
ClinicalAnalysisService
     ↑
Triage Engine
     ↑
Safety Gate
     ↑
FastAPI
     ↑
Frontend
```

This allows the frontend and API to evolve without duplicating or modifying the core ML implementation.

---

# 🚦 Quick Start

For developers who already have the repository:

### Terminal 1 — Backend

```cmd
cd backend
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

### Terminal 2 — Frontend

```cmd
cd frontend
npm install
npm run dev
```

Then open the frontend development URL shown by Vite.

Backend API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 🤝 Contribution

When contributing:

1. Create a feature branch.

```cmd
git switch -c feature/<feature-name>
```

2. Make your changes.

3. Run tests.

```cmd
python -m pytest
```

4. Check Git status.

```cmd
git status
```

5. Commit:

```cmd
git add .
git commit -m "Describe your change"
```

6. Push:

```cmd
git push -u origin feature/<feature-name>
```

7. Create a Pull Request for review.

---

# 📄 License

Add the project's chosen license here before public release.

---

# ⚠️ Disclaimer

This system is a **research/prototype clinical intelligence application**.

It is not intended to:

* Replace doctors
* Provide definitive medical diagnoses
* Replace emergency services
* Independently prescribe medication
* Replace professional clinical judgment

Clinical decisions should be made by qualified healthcare professionals using appropriate clinical information and established medical protocols.

---

## ❤️ Project Goal

The long-term goal is to build a safe and intelligent clinical intelligence platform that can help identify important health risks earlier, organize clinical information, support treatment decisions, and route appropriate cases to healthcare professionals.

**AI assists the clinician — it does not replace the clinician.**
