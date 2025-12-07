# 📘 **FBRA: Forward–Backward Reachability Analysis (Reimplementation)**
*A pure-Python verification framework for neural network–controlled systems.*

---

## 🚀 Overview

This repository contains a full re-implementation of the **FBRA (Forward–Backward Reachability Analysis)** algorithm for verifying safety properties of **neural network–controlled systems** (NNCS).  
The implementation supports several benchmark systems:

- **Ground Robot (Safe)**
- **Ground Robot (Buggy / Unsafe)**
- **Double Integrator**
- **Quadrotor (Simplified 6D Hover Model)**

The goal of the project is to reproduce and experiment with the verification pipeline presented in the paper:

> *Verifying Neural Network Controlled Systems by Combining Forward and Backward Reachability Analysis*  

Core features include:

- Interval Bound Propagation (IBP) for NNs  
- Box-based forward reachability  
- Sampling-based backward refinement  
- Combined FBRA verification  
- Modularity for adding new systems and controllers  

---

## 📂 Project Structure

```
fbra_reimplementation/
│
├── main.py
├── requirements.txt
│
├── fbra/                       # Core verification library
│   ├── boxes.py
│   ├── forward.py
│   ├── backward.py
│   ├── refine_forward.py
│   ├── nn_bounds.py
│   ├── verifier.py
│
├── experiments/               # All experiment-specific code
│   ├── controller.py          # Safe, Buggy, Double Integrator, Quadrotor controllers
│   ├── dynamics.py            # Dynamics for all benchmarks
│   ├── sets.py                # Initial & unsafe sets
│
├── utils/
│   ├── merge.py
│   ├── sampling.py
│   ├── visualization.py
│
└── scripts/                   # Executable experiment scripts
    ├── run_ground_robot.py
    ├── run_ground_robot_buggy.py
    ├── run_double_integrator.py
    ├── run_quadrotor.py
```

---

## 🧩 Supported Benchmarks

### **1. Ground Robot (Safe)**  
- 2D state space  
- 2D control  
- Expected result: **Safe**

### **2. Ground Robot (Buggy)**  
- Same model as above  
- Controller intentionally biased toward unsafe region  
- Expected result: **Unsafe**

### **3. Double Integrator**  
- Classic 2D system  
- Expected result: **Safe**

### **4. Quadrotor (Simplified 6-D Hover Model)**  
- State: position + velocity (6D)  
- Control: accelerations (3D)  
- Linearized near-hover dynamics  
- Expected: Often **Unknown** or **Unsafe** due to box over-approximation

---

## 🛠️ Installation

### **1. Clone the repository**
```sh
git clone https://github.com/pratox1112/FBRA_ReImplementation.git
cd fbra_reimplementation
```

### **2. Create virtual environment**
```sh
python -m venv venv
```

### **3. Activate environment**
**Windows (PowerShell):**
```sh
venv\Scripts\Activate.ps1
```

**CMD:**
```sh
venv\Scripts\activate.bat
```

### **4. Install dependencies**
```sh
pip install -r requirements.txt
```

---

## ▶️ Running Experiments

All experiments must be run from the **project root**:

### **Ground Robot (Safe)**
```sh
python scripts/run_ground_robot.py
```

### **Ground Robot (Buggy)**
```sh
python scripts/run_ground_robot_buggy.py
```

### **Double Integrator**
```sh
python scripts/run_double_integrator.py
```

### **Quadrotor**
```sh
python scripts/run_quadrotor.py
```

---

## 📊 Visualizing Reachable Sets

A helper visualization module is included:

```
utils/visualization.py
```

You can integrate it into any script to generate:

- Forward-only reachable sets  
- FBRA refinement steps  
- Final verified sets  

Example usage:

```python
from utils.visualization import plot_reachable_sets
plot_reachable_sets(R_f, unsafe_region, "Reachability Plot")
```

---

## 🧠 Extending the Framework

You can easily add new systems:

1. Add a new controller in `experiments/controller.py`
2. Add new system dynamics in `experiments/dynamics.py`
3. Add initial/unsafe sets in `experiments/sets.py`
4. Create a new script in `scripts/`

The FBRA algorithm (`fbra/verifier.py`) does not need any modification.

---

## ⚠️ Notes and Limitations

- The current IBP implementation only supports **Linear + ReLU** activations.
- Box-based reachability scales poorly in high dimensions (e.g., quadrotor).
- A full nonlinear quadrotor model is **not recommended** without zonotopes or advanced bounding.

---

## 👤 Author

Pratosh Karthikeyan
Ahilesh Vadivel


