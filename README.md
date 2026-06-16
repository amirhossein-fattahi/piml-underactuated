# piml-underactuated 🤖⚙️

Hello! This is the official repository for my Master's thesis project in Control Systems Engineering at the University of Padua. 

The main goal of this project is to compare physics-informed machine learning (PiML) models with standard "black-box" neural networks. Specifically, I am looking at how well these models learn the dynamics of underactuated robots, and how useful they are for downstream tasks like energy-based swing-up control.

Currently, the focus is on two classic systems:
* **Acrobot**
* **Pendubot**

*(I might extend this to other underactuated robots later, but for now, the benchmark is built around these two!)*

## Why this project?
Standard machine learning models (like normal MLPs) are powerful, but they don't always respect real-world physics. Over long rollout horizons, they can "drift" and violate basic rules like the conservation of energy. 

In this research, I am using methods like Deep Lagrangian Networks (DeLaN) to embed physics priors directly into the learning process. This benchmark study maps out exactly when physics-informed models actually help compared to vanilla models. I evaluate them across four main axes:
1.  One-step prediction accuracy
2.  Long-horizon rollout stability
3.  Energy drift over time
4.  Downstream control performance (energy-based swing-up)

## Repository Structure
To make things modular and easy to read, the code is structured like this:

* `configs/`: YAML files to manage experiment parameters easily.
* `src/envs/`: The true physics simulators for the Acrobot and Pendubot.
* `src/models/`: The "Model Zoo" built in **PyTorch**, containing both baselines (e.g., vanilla MLP) and physics-informed models (e.g., LNN).
* `src/controllers/`: Energy-based controllers used to test how well the learned dynamics perform in practice.
* `scripts/`: The main files you need to run to generate data, train models, and evaluate them.
* `notebooks/`: Jupyter notebooks for data visualization and plotting the energy drift.

## Installation & Setup

This project uses standard Python `venv` for environment management. 

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/piml-underactuated.git](https://github.com/yourusername/piml-underactuated.git)
cd piml-underactuated
```
**2. Create and activate the virtual environment**
```bash
python -m venv venv
```
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate

**3. Install dependencies**
First, install PyTorch (adjust the CUDA version based on your hardware if you are using a GPU):
```bash
pip install torch torchvision torchaudio
```

Then, install the rest of the project requirements:
```bash
pip install -r requirements.txt
```

## How to Run

```bash
# Example of how to run a training experiment:
python scripts/02_train.py model=lnn env=acrobot
```

## Contact
If you have any questions about the code, the implementations, or control systems in general, feel free to reach out or open an issue!
