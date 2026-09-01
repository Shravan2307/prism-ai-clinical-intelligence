# 🏥 Medical AI Clinical Intelligence — How to Use

This guide explains how to run and use the Medical AI Clinical Intelligence application locally.

---

## 1. Prerequisites

Make sure you have installed:

* Python 3.11+
* Node.js
* npm
* Git

Check the installations:

```bash
python --version
node --version
npm --version
git --version
```

---

# 2. Project Setup

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project:

```bash
cd prism-ai-clinical-intelligence
```

The project contains:

```text
prism-ai-clinical-intelligence/
├── backend/
├── frontend/
├── README.md
└── HOW_TO_USE.md
```

---

# 3. Start the Backend

Open Terminal 1.

```bash
cd backend
```

## Create a virtual environment

Windows:

```cmd
python -m venv venv
```

Activate it:

```cmd
venv\Scripts\activate
```

You should now see:

```text
(venv)
```

in your terminal.

---

## Install backend dependencies

```cmd
pip install -r requirements.txt
```

---

## Configure environment variables

If the project requires environment variables, create a `.env` file inside the backend/project root according to `.env.example`.

Example:

```env
APP_ENV=development
FRONTEND_URL=http://localhost:5173
```

Never add real API keys or secrets to GitHub.

---

# 4. Run FastAPI

From the `backend` directory:

```cmd
python -m uvicorn app.main:app --reload
```

You should see something similar to:

```text
Uvicorn running on http://127.0.0.1:8000
```

The backend is now running.

### API

```text
http://127.0.0.1:8000
```

### Swagger API Documentation

Open:

```text
http://127.0.0.1:8000/docs
```

### ReDoc

Open:

```text
http://127.0.0.1:8000/redoc
```

---

# 5. Start the Frontend

Open **Terminal 2**.

Move into the frontend directory:

```cmd
cd frontend
```

Before installing dependencies, make sure this directory contains:

```text
package.json
```

Check:

```cmd
dir package.json
```

If `package.json` exists:

```cmd
npm install
```

Then start the frontend:

```cmd
npm run dev
```

The terminal will show the frontend URL, normally something similar to:

```text
http://localhost:5173
```

Open that URL in your browser.

---

# 6. Configure Frontend → Backend Connection

The frontend communicates with the FastAPI backend through HTTP.

For a Vite frontend, configure:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

The frontend API client should use this variable.

The architecture is:

```text
Browser
   │
   ▼
Frontend
   │
   │ HTTP/JSON
   ▼
FastAPI
   │
   ▼
PatientRequest
   │
   ▼
Safety Gate
   │
   ▼
Triage Engine
   │
   ▼
ClinicalAnalysisService
   │
   ▼
Existing ML / AI Components
   │
   ▼
Clinical Response
   │
   ▼
Frontend Dashboard
```

---

# 7. Using the Application

Once both servers are running:

### Step 1 — Open the frontend

Open the URL provided by the frontend development server.

Example:

```text
http://localhost:5173
```

---

### Step 2 — Enter Patient Information

Enter the information requested by the application.

The frontend should only request fields required by the backend `PatientRequest` model.

Do not enter unnecessary information.

---

### Step 3 — Submit the Analysis

Click the analysis button.

For example:

```text
Analyze Patient
```

The frontend sends the patient information to FastAPI.

---

### Step 4 — Backend Processing

The backend processes the request through the clinical workflow:

```text
PatientRequest
      ↓
Safety Gate
      ↓
Triage Engine
      ↓
ClinicalAnalysisService
      ↓
Existing ML Components
      ↓
Structured Clinical Response
```

---

### Step 5 — View Results

The frontend displays the response returned by the backend.

Depending on the backend result, the interface may display:

* Clinical assessment
* Risk information
* Contributing factors
* Recommended next step
* Treatment/recommendation information
* Clinical explanation
* Triage status
* Safety status

---

# 8. Safety / Urgent Cases

The Safety Gate is controlled by the backend.

If the backend returns an urgent or emergency result, the frontend should clearly show the escalation.

Example:

```text
URGENT MEDICAL REVIEW REQUIRED
```

The application can then route the case toward the appropriate doctor workflow.

### Important

The frontend does NOT calculate:

* Medical risk
* Emergency status
* Disease probability
* Treatment ranking
* Clinical reasoning

These decisions come from the backend.

---

# 9. Doctor Workflow

Cases requiring professional review can be displayed in the doctor dashboard.

A doctor can view backend-provided information such as:

```text
Patient
   ↓
Safety Status
   ↓
Triage Status
   ↓
Clinical Findings
   ↓
Risk Information
   ↓
Contributing Factors
   ↓
Recommendations
```

The frontend should never change or override the backend's clinical decision.

---

# 10. Testing the Backend Without the Frontend

You can test the API directly through Swagger.

Start FastAPI:

```cmd
python -m uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Find the clinical analysis endpoint.

Click:

```text
Try it out
```

Enter a valid `PatientRequest`.

Then click:

```text
Execute
```

Swagger will display the actual backend response.

This is useful for verifying the backend before debugging frontend integration.

---

# 11. Running Tests

Open a terminal in the backend:

```cmd
cd backend
```

Activate the virtual environment:

```cmd
venv\Scripts\activate
```

Run:

```cmd
python -m pytest
```

For detailed output:

```cmd
python -m pytest -v
```

All existing tests should continue to pass after frontend/application-layer integration.

---

# 12. Troubleshooting

## `npm install` says package.json does not exist

Example:

```text
npm ERR! enoent Could not read package.json
```

This means you are not inside the actual frontend project directory.

From the project root run:

```cmd
dir /s /b package.json
```

Find the frontend's `package.json`.

Then move into that directory:

```cmd
cd <frontend-directory>
```

and run:

```cmd
npm install
npm run dev
```

---

## Frontend cannot connect to backend

Check that FastAPI is running:

```text
http://127.0.0.1:8000/docs
```

Then check the frontend environment variable:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Restart the frontend after changing `.env`:

```cmd
npm run dev
```

---

## CORS Error

If the browser reports a CORS error, verify that FastAPI allows the frontend development origin.

Typical development frontend:

```text
http://localhost:5173
```

The backend should be configured to allow the appropriate frontend origin.

Do not use unrestricted CORS configuration for production without understanding the security implications.

---

## Backend does not start

Check:

```cmd
python --version
```

Activate the virtual environment:

```cmd
venv\Scripts\activate
```

Install dependencies again:

```cmd
pip install -r requirements.txt
```

Then:

```cmd
python -m uvicorn app.main:app --reload
```

---

# 13. Recommended Development Workflow

When developing a new feature:

### 1. Start backend

```cmd
cd backend
venv\Scripts\activate
python -m uvicorn app.main:app --reload
```

### 2. Start frontend

Open another terminal:

```cmd
cd frontend
npm run dev
```

### 3. Test backend

Open:

```text
http://127.0.0.1:8000/docs
```

### 4. Test frontend

Open the frontend URL.

### 5. Run automated tests

```cmd
python -m pytest
```

### 6. Check Git

```cmd
git status
```

---

# 14. Development Rules

When modifying this project, follow these rules.

## Reuse Existing ML

The existing ML implementation is the source of truth.

Do not duplicate it.

## Do Not Create Another Risk Engine

Risk calculations must remain in the existing backend implementation.

## Do Not Create Another Treatment Ranking Engine

Treatment ranking must remain in the existing implementation.

## Do Not Create Another Causal Engine

Causal analysis must remain in the existing implementation.

## Do Not Put Clinical Logic in React

React should handle:

```text
Input
↓
API Request
↓
Response
↓
UI
```

The backend handles:

```text
Validation
↓
Safety
↓
Triage
↓
ML
↓
Clinical Analysis
```

---

# 15. Production Concept

The development architecture is:

```text
localhost:5173
       │
       │ HTTP
       ▼
localhost:8000
       │
       ▼
FastAPI
       │
       ▼
ClinicalAnalysisService
       │
       ▼
ML / AI
```

In production, these URLs will be replaced with the deployed frontend and backend services.

Environment variables should be used so the frontend does not need source-code changes when the backend URL changes.

---

# 16. Quick Start — Copy/Paste

### Backend

```cmd
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

### Frontend

Open another terminal:

```cmd
cd frontend
npm install
npm run dev
```

Then open the frontend URL displayed in the terminal.

---

# 17. Final Application Flow

```text
                 USER
                  │
                  ▼
          ┌───────────────┐
          │ Patient Form  │
          └───────┬───────┘
                  │
                  ▼
          ┌───────────────┐
          │    FastAPI    │
          └───────┬───────┘
                  │
                  ▼
          ┌───────────────┐
          │ PatientRequest│
          └───────┬───────┘
                  │
                  ▼
          ┌───────────────┐
          │  Safety Gate  │
          └───────┬───────┘
                  │
                  ▼
          ┌───────────────┐
          │ Triage Engine │
          └───────┬───────┘
                  │
                  ▼
       ┌──────────────────────┐
       │ ClinicalAnalysis     │
       │ Service              │
       └──────────┬───────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │ Existing ML / AI     │
       │ Implementation       │
       └──────────┬───────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │ Clinical Response    │
       └──────────┬───────────┘
                  │
          ┌───────┴────────┐
          ▼                ▼
     Patient UI       Doctor Workflow
```

---

# ⚠️ Medical Disclaimer

This project is a research and prototype clinical intelligence system.

It is not a substitute for:

* Professional medical diagnosis
* Emergency medical services
* Physician judgment
* Clinical examination
* Established medical protocols

Any real-world clinical deployment requires appropriate validation, regulatory compliance, security controls, clinical oversight, and professional review.
