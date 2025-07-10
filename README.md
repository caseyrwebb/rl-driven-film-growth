# RL Driven Film Growth

A minimal digital twin for a Chemical Vapor Deposition (CVD) reactor with reinforcement learning (RL) optimization.  
This project demonstrates how to control and optimize CVD film growth using a simple Arrhenius-based process model and a Deep Q-Network (DQN) agent.

---

## 🚀 Overview

This repository provides:

- A **simulated CVD reactor environment** using an Arrhenius kinetic model (temperature and precursor flow rate affect deposition).
- A **Gymnasium-compatible environment** for RL experiments.
- Training and evaluation scripts using Stable-Baselines3’s DQN agent.
- Utilities for **visualizing film thickness, temperature, and flow trajectories**.
- Notebooks for interactive exploration and results analysis.

---

## 📈 Why This Project?

CVD processes are fundamental in semiconductor manufacturing, but optimizing them is challenging due to nonlinear chemical dynamics and tight process constraints.  
This project creates a **testbed for RL-based process control**—useful as a teaching tool, a starting point for research, or a basis for scaling up to more complex digital twins (e.g., with Cantera or real fab data).

---

## 🧠 Features

- **Arrhenius-based CVD Environment**:  
  Simple, tunable model for film deposition as a function of temperature and gas flow rate.

- **Reinforcement Learning Control**:  
  Out-of-the-box DQN agent learns to reach a target film thickness efficiently.

- **Reproducible Experiments**:  
  Jupyter notebook and Python scripts for random-policy baselines, agent training, and evaluation.

- **Visualization Utilities**:  
  Plot trajectories of thickness, temperature, and flow rate for intuitive understanding and debugging.

- **Modular Design**:  
  Well-structured Python package, ready for extension (e.g., more realistic kinetics, pressure control, or other RL algorithms).

---

## 🗂️ Project Structure

```
rl-driven-film-growth/
├── README.md
├── notebooks/  # Jupyter notebooks for exploration and analysis
├── pyproject.toml
├── src/
│   ├── agents/  # RL agent training & evaluation scripts
│   ├── cvd_env/  # CVD reactor environment code
│   └── utils/  # Plotting and helper functions
├── tests/  # Unit tests for the environment
```

---

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone git@github.com:caseyrwebb/rl-driven-film-growth.git
   cd rl-driven-film-growth
   ```
2. **Create and activate a virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # or
   .venv\Scripts\activate     # Windows
   ```

3. **Install the package and development dependencies:**

   ```bash
   pip install -e ".[dev]"
   ```

   Setup pre-commit hooks to ensure code quality:

   ```bash
   pre-commit install
   ```

4. **(Optional) Install Jupyter kernel for the environment:**
   ```bash
   python -m ipykernel install --user --name=rl-driven-film-growth --display-name="RL Driven Film Growth"
   ```

## 🧪 Running the Environment

You can run a random policy simulation from the command line:

```bash
python src/cvd_env/reactor_env.py
```

Or train a DQN agent:

```bash
python src/agents/dqn_trainer.py
```

## 📓 Interactive Analysis

Open the main notebook for exploration:

```bash
jupyter notebook notebooks/explore_env.ipynb
```

- Compare random actions vs. RL agent

- Visualize learning curves and control trajectories

- Experiment with different reward structures and hyperparameters

## 🧑‍💻 Testing

Run unit tests with pytest:

```bash
pytest
```

## 🔧 Customization

- Modify the kinetic model:
  Edit `src/cvd_env/reactor_env.py` to change Arrhenius parameters, add pressure, or introduce more complex dynamics.

- Try other RL agents:
  Swap in other Stable-Baselines3 agents (e.g., PPO, A2C) in `src/agents/dqn_trainer.py`.

## 📚 References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
  Library for creating and using RL environments.
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/)
  Collection of RL agents built on top of Gymnasium.
- [NIST Chemical Kinetics Database](https://kinetics.nist.gov/kinetics/index.jsp) Database of reaction rate constants and kinetics for thousands of reactions.
- [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/) Chemical and physical property data for a wide range of species.
