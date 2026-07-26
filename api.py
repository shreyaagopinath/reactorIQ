from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from scipy.integrate import odeint

app = FastAPI(title="ReactorIQ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model
class CSTRInput(BaseModel):
    concentration_in: float = 1.0
    flow_rate: float = 1.0
    volume: float = 100.0
    rate_constant: float = 0.1
    temperature: float = 350.0

# Output model
class SimulationResult(BaseModel):
    conversion: float
    concentration_out: float
    residence_time: float
    reactor_type: str

@app.get("/")
def root():
    return {"message": "ReactorIQ API", "version": "1.0"}

@app.post("/simulate/cstr", response_model=SimulationResult)
def simulate_cstr(params: CSTRInput):
    # Residence time
    tau = params.volume / params.flow_rate
    
    # CSTR steady state: Ca = Ca0 / (1 + k*tau)
    concentration_out = params.concentration_in / (
        1 + params.rate_constant * tau
    )
    
    conversion = (
        (params.concentration_in - concentration_out) 
        / params.concentration_in
    )
    
    return SimulationResult(
        conversion=round(conversion, 4),
        concentration_out=round(concentration_out, 4),
        residence_time=round(tau, 2),
        reactor_type="CSTR"
    )

@app.get("/simulate/history")
def get_history():
    return {"message": "History endpoint — connect to PostgreSQL in v2"}