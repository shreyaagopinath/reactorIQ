# ReactorIQ

A chemical reactor design and analysis application built independently in Python.
ReactorIQ models CSTR, PFR, Batch, and Packed Bed reactors, solving the 
governing equations numerically to generate concentration profiles, conversion 
curves, temperature gradients, and reactor sizing analysis.


---

## Overview

ReactorIQ was built as an independent study tool to reinforce and apply 
chemical engineering fundamentals — specifically reaction kinetics, 
thermodynamics, and reactor design. Given user-defined feed conditions, 
kinetic parameters, and reactor configuration, the application solves the 
relevant ODEs or algebraic equations and visualizes the results interactively.

The project required implementing the underlying ChemE concepts from first 
principles: deriving rate expressions, setting up mole and energy balances, 
and applying numerical methods to solve systems that have no closed-form 
solution.

---

## Reactor Models

| Reactor | Method | Independent Variable |
|---|---|---|
| CSTR | Steady-state nonlinear solver (`scipy.optimize.fsolve`) | Residence time τ |
| PFR | ODE integration (`scipy.integrate.solve_ivp`, RK45) | Reactor volume V |
| Batch | ODE integration (`scipy.integrate.solve_ivp`, RK45) | Time t |
| Packed Bed (PBR) | ODE integration (`scipy.integrate.solve_ivp`, RK45) | Catalyst weight W |

---

## Features

**Kinetics**
- 1st order, 2nd order, and bimolecular (A + B → P) rate laws
- Irreversible and reversible reactions
- Arrhenius temperature correction: `k(T) = k·exp(−Eₐ/R · (1/T − 1/T_ref))`

**Energy Balance**
- Isothermal or adiabatic operation
- Coupled temperature ODE: `dT/dV = −ΔH_rxn · r / ρCp`
- Adiabatic temperature rise profile for PFR, Batch, and PBR

**Outputs**
- Concentration profiles for reactant (Cₐ) and product (C_B)
- Fractional conversion X along reactor volume or time
- Temperature profiles for adiabatic operation
- Levenspiel plots (required PFR volume vs. target conversion)
- CSTR sizing curve (conversion vs. residence time/volume)

---

## Screenshots

<img width="1725" height="912" alt="Screenshot 2026-04-09 at 7 51 27 PM" src="https://github.com/user-attachments/assets/d5c3420e-c4c0-4aff-91f8-72ce7cb1dc94" />

<img width="1707" height="827" alt="Screenshot 2026-04-09 at 7 53 01 PM" src="https://github.com/user-attachments/assets/d9d13e3e-630d-4df6-a4e6-0b4164a09a7f" />

<img width="1702" height="873" alt="Screenshot 2026-04-09 at 7 56 59 PM" src="https://github.com/user-attachments/assets/dc207756-be54-4ccc-9630-3ee6f93a3edd" />

<img width="1687" height="853" alt="Screenshot 2026-04-09 at 7 58 36 PM" src="https://github.com/user-attachments/assets/49b0e22f-8e90-4ac2-b2b7-172af9d24428" />





---

## How to Run

```bash
pip install streamlit numpy scipy plotly
streamlit run app.py
