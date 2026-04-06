# 🐇 Razzy-Rabbit Robotics Suite

Razzy-Rabbit is a Linux-native, Python-based Engineering Suite designed for the rapid prototyping 
and Finite Element Analysis (FEA) of robotic structures. 
Razzy-Rabbit is built for automation, allowing developers to bridge the gap between 3D design 
and real-world mechanical validation.

## 🎯 Project Goals
- Physics-Driven Design: Integrated GMSH (meshing) and CalculiX (solving) to validate robot parts before manufacturing.
- Multi-Workspace Workflow: Manage multiple independent design tabs simultaneously.

## 🛠 Prerequisites
1. System Dependencies (Linux/Ubuntu)

You must have the physical solvers installed on your host machine to handle the heavy math:
Bash

### Install the CalculiX Solver
```bash
sudo apt update && sudo apt install calculix-ccx
```

### Ensure system has GL libraries for PyVista/VTK
```bash
sudo apt install libglu1-mesa
```

2. Python Environment

This project uses uv for fast, reproducible dependency management and a src layout for clean packaging.
## 🚀 Getting Started
Installation

Clone the repository and sync the environment using uv:
```bash
git clone https://github.com/yourusername/razzy-rabit.git
cd razzy-rabit
uv sync
```

## Running the Application

To run the suite with GPU acceleration and the modular package structure:
Bash

### Launch the Gui Launcher
```bash
QT_QPA_PLATFORM=xcb DRI_PRIME=1 uv run python -m razzy_rabbit
```

### 🕹 Mechanical Workflow
- Generate Geometry: Use 📁 Workspace -> 🛠 Samples to create a Robotic Link or Bracket.
- Generate Volume Mesh: Navigate to 🔬 Analysis -> Generate Mesh (GMSH). This fills the hollow STL skin with a solid tetrahedral skeleton.
- Define Physics (Picking):
- Select Mark Fixed Support.
    -  Left-Click vertices to add Red Squares (Constraints).
    - Left-Click an existing square to Unselect it.
    - Solve: The suite translates these markers into a CalculiX .inp file to solve the [K]u=f equilibrium equation.

#### 🗺 Future Directions
- Post-Processing: Automated displacement and Von Mises Stress visualization directly in the 3D viewport.

#### 📚 Documentation & Theory
- For developers interested in the underlying mathematics:
- CalculiX Manual: The "Bible" for the CCX solver (dhondt.de).
- GMSH Reference: Finite element mesh generation (gmsh.info).
- PyVista: Professional 3D visualization and mesh manipulation (docs.pyvista.org).
