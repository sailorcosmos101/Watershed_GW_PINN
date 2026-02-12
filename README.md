# Watershed_GW_PINN

This repository provides the official implementation of the Physics-Informed Neural Network (PINN) for simulating 2D steady-state groundwater flow in watershed-scale confined aquifers.

### Abstract
The applicability of Physics-Informed Neural Network (PINN) was investigated for watershed-scale groundwater modeling. A PINN-based 2D steady-state confined aquifer model was developed, capable of implementing hydrological components such as watershed boundaries, recharge, and baseflow as essential boundary conditions. The PINN was trained to minimize a loss function that incorporates both Darcy’s law and the boundary conditions. The PINN-based groundwater model was verified by comparison with numerical models for the same hydrogeological conditions. The model accurately reflected watershed boundaries with no-flow and constant-head conditions, demonstrating high predictive accuracy and excellent agreement with the numerical model. The model simulated recharge and baseflow effectively, confirming its ability to implement the complex and diverse hydraulic conditions required for building watershed-scale models. The model demonstrated its applicability to large-scale domains, which was enabled by scaling techniques. The applicability to island and general watersheds was confirmed by verifying its ability to learn from various polygonal domains.

## 1. How to Run
### Environment Setup (Versions of package are mentioned in requirements.txt file)
- Framework: JAX
- Optimization: Optax
- Visualization: Matplotlib, NumPy, Pandas
Inputs & Parameters
- Domain Geometry: Define watershed boundaries (polygonal domains)
- Hydraulic Parameters: Transmissivity, Recharge rates, and constant head at river
- Scaling Factors: Domain-specific scaling to normalize physical coordinates (0 to 10,000 m) into the PINN-friendly range (0 to 1)
- Reference Data: MODFLOW-generated text files for accuracy verification (RMSE calculation)

### Workflow
1) generate collocation points
2) construct loss function for each case
3) training neural network
4) Visualization and verification

### Conceptual model for each cases
Watershed_GW_PINN models are developed for 6 cases reflecting various hydrologic processes.
<img width="940" height="493" alt="image" src="https://github.com/user-attachments/assets/ac2a3b9f-29cd-4fba-aea0-584c67c26b70" />\
<img width="939" height="483" alt="image" src="https://github.com/user-attachments/assets/8e78f93e-8b97-4be7-871f-67d47866bf95" />

## 2. File Structure & Descriptions
requirements.txt – required package and versions

notebooks/ - jupyter notebeook (.ipynb) file which contains PINN for all cases

data/ - Directory containing MODFLOW reference text files for all cases
