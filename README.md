# 🚀 AeroForge — Drone Simulation & Reinforcement Learning Framework

AeroForge is a full-stack research platform for reinforcement-learning-based autonomous drones, built on:

- **C++ high-performance physics** (Bullet)
- **Python bindings (pybind11)** to interface the engine with RL
- **Custom Gymnasium environments**
- **Stable-Baselines3 PPO training pipelines**
- A **modular multi-stage roadmap**: hover → navigation → trajectory tracking → real hardware

AeroForge aims to be a professional, scalable foundation for robotics research, simulation, and sim-to-real transfer.

---

# 📦 Project Structure

```text
AeroForge/
│
├── CMakeLists.txt
├── external/                 # Bullet, pybind11
├── include/
├── src/
│   └── aeroforge_sim/        # C++ SimCore physics engine
│
├── build/                    # Compiled artifacts (ignored by git)
│
├── python/
│   └── aeroforge/
│       ├── core/
│       │   └── simcore_loader.py
│       ├── envs/
│       │   └── drone_nav_env.py
│       └── scripts/
│           └── train_hover_ppo.py
│
└── tests/
    └── python/
        └── test_drone_nav_env.py
```

---

# ⚙️ Installation Guide

## 1. Clone the repository
```bash
git clone https://github.com/<your-username>/AeroForge.git
cd AeroForge
```

---

## 2. Configure & Build the Project
```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

---

## 3. Install Python Dependencies
```bash
pip install -r python/requirements.txt
# or
pip install numpy gymnasium stable-baselines3 pytest
```

---

## 4. Verify Python Bindings
```python
import aeroforge_core
sim = aeroforge_core.SimCore()
sim.initialize()
sim.reset()
print(sim.get_observation())
```

---

# 🤖 C++ Simulation Core

AeroForge uses **Bullet Physics** to simulate:

- Drone 6-DoF rigid body  
- Ground plane collision  
- Integration step exported to Python

Raw observation (13D):
```text
[pos(3), quat(4), lin_vel(3), ang_vel(3)]
```

Python bindings live in:
```text
build/src/bindings/python/aeroforge_core.so
```

---

# 🧠 Python Environment — DroneNavEnv

### Observation (8D):
```text
[z, vz, dz, roll, pitch, p, q, r]
```

### Action (4D):
```text
[thrust_cmd, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd]
```

### Reward combines:
- Altitude error  
- Vertical velocity  
- Tilt  
- Angular rates  
- Progress reward  
- Hover bonus  

### Episode ends when:
- max_steps reached  
- z < 0.1  
- z > 50  

---

# 🧪 Unit Tests

Run tests:
```bash
pytest -q
```

Specific test:
```bash
pytest tests/python/test_drone_nav_env.py -q
```

CTest:
```bash
ctest -R python_drone_nav_env --output-on-failure
```

---

# 🎯 Training PPO — Hover Task

Run training:
```bash
python python/aeroforge/scripts/train_hover_ppo.py
```

Model saved in:
```text
models/hover_ppo/ppo_drone_hover.zip
```

---

# 🚁 Evaluation Logging

Evaluation prints:
- z, vz, dz  
- roll, pitch  
- p, q, r  
- reward & cumulative reward  
- distance to target  

Example:
```text
[Eval] Episode 3 | Step 87
  • Reward this step     : -0.4998
  • Observation          : z=1.043, dz=0.043
  • Distance to target   : 0.0431
```

---

# 📘 Development Stages (Roadmap)

✔ **Stage 1 — Bullet Physics**  
✔ **Stage 2 — Python Bindings**  
✔ **Stage 3 — Basic RL Env**  
✔ **Stage 4 — Hover RL**  

🔜 **Stage 5 — Full 3D Navigation**  
🔜 **Stage 6 — Trajectory Following**  
🔜 **Stage 7 — Sensor & Noise**  
🔜 **Stage 8 — PID / MPC Control**  
🔜 **Stage 9 — Real Hardware Integration**  

---

# 🛠 Add README to Git

```bash
git add README.md
git commit -m "Add project README"
git push origin main
```
