# ReactorIQ

A deployed chemical reactor simulation platform with a REST API backend and interactive Streamlit frontend.

**Live app:** https://reactoriq.streamlit.app

## Architecture

Streamlit Frontend (app.py)
↓ HTTP POST
FastAPI Backend (api.py)
↓ SciPy ODE solvers
Simulation Engine

## API Endpoints

- `POST /simulate/cstr` — Continuous Stirred Tank Reactor simulation
- `POST /simulate/pfr` — Plug Flow Reactor simulation
- `GET /simulate/history` — Simulation run history (PostgreSQL in v2)

## Why this architecture

The simulation engine is separated from the UI so any frontend — React, mobile, or another service — can consume the same calculations. The Streamlit app calls the API rather than running math directly.

## Running locally

```bash
# Start the API
uvicorn api:app --reload

# Start the frontend (separate terminal)
streamlit run app.py
```

## What I'd build next

- PostgreSQL backend to persist simulation runs
- JWT auth so users can retrieve their history
- Natural language interface using Claude API — describe what you want to model in plain English
- Docker containerization for deployment

