# Fault-Tolerant Roll Stabilization of a CMG-Actuated Motorcycle

**Robust EKF–MPC vs. Distilled Neural MPC vs. Residual Reinforcement Learning**

This repository contains the research implementation and experimental materials for fault-tolerant roll stabilization of a motorcycle equipped with Control Moment Gyroscopes (CMGs).

The project investigates and compares model-based, learning-based, and hybrid control strategies for maintaining motorcycle stability under nominal and perturbed operating conditions.

## Research Objectives

The primary objective is to investigate robust and adaptive control strategies for motorcycle roll stabilization, with particular emphasis on fault tolerance, control performance, and computational efficiency.

The study focuses on three complementary approaches:

1. **Robust EKF–MPC:** A model-based control framework combining state estimation using an Extended Kalman Filter (EKF) with Model Predictive Control (MPC).
2. **Distilled Neural MPC:** A learning-based approximation of MPC designed to reproduce its control behavior while reducing computational complexity.
3. **Residual Reinforcement Learning:** A hybrid control strategy that combines a conventional control framework with a reinforcement learning-based residual policy to improve adaptability.

## System Description

The motorcycle stabilization system employs Control Moment Gyroscopes (CMGs) to generate corrective torques for roll stabilization.

The control framework considers the motorcycle's roll dynamics, steering-related states, and CMG actuation.

The simulation environment is developed using Python and MuJoCo.

## Repository Structure

```text
motorcycle-ft-roll-stabilization/
│
├── 00_paper/              # Research manuscript
├── 01_code_core/          # Core implementation and control algorithms
├── 02_mujoco/             # MuJoCo simulation environment
├── 03_checkpoints/        # Model checkpoints
├── 04_data/               # Experimental and simulation datasets
├── 05_results/            # Main results and figures
├── 05_results_folders/    # Additional result materials
├── 06_provenance/         # Experiment provenance and reproducibility records
│
├── COPY_LOG.txt           # File transfer and copy records
├── MANIFEST_SHA256.csv    # File integrity manifest
├── README.md
└── .gitignore
```

Large experiment run directories are intentionally excluded from this repository to keep the public release manageable.

## Technologies

- Python
- MuJoCo
- NumPy
- SciPy
- PyTorch
- Model Predictive Control (MPC)
- Extended Kalman Filtering (EKF)
- Deep Reinforcement Learning

## Reproducibility

The repository includes the core implementation, simulation environment, selected datasets, model checkpoints, and research results.

The provenance records and SHA-256 manifest are provided to support file integrity verification and experimental traceability.

Specific execution commands and dependency versions should be taken from the corresponding project configuration and implementation files.

## Research Focus

This work explores the intersection of:

- Fault-Tolerant Control
- Nonlinear and Predictive Control
- State Estimation
- Physics-Based Modeling
- Neural Network-Based Control
- Reinforcement Learning
- Motorcycle Dynamics and Stabilization

## Author

**Moein Razavi-Nabavi**

M.Sc. in Mechanical Engineering – Control Systems

Iran University of Science and Technology

**Research Interests:** Fault-Tolerant Control, Autonomous Systems, Reinforcement Learning, Model Predictive Control, and Intelligent Control Systems.

---

*This repository is intended for research, academic evaluation, and further development of advanced control strategies for motorcycle stabilization.*