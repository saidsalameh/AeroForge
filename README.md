# 🚀 AeroForge — Drone Simulation & Reinforcement Learning Framework

AeroForge is a full-stack drone simulation and reinforcement learning framework combining:

- **High-performance C++ physics** using Bullet
- **Python bindings (pybind11)** for control & RL
- **Custom RL environments** for Gymnasium
- **Stable-Baselines3 PPO training pipelines**
- **Modular roadmap** from basic hover → full 3D navigation → trajectory tracking → real hardware deployment

The goal of AeroForge is to provide a professional, scalable foundation for autonomous drone RL research, simulation, and real-world transfer.

---

# 📦 Project Structure

AeroForge/
│
├── CMakeLists.txt
├── external/ # Bullet, pybind11
├── include/
├── src/
│ └── aeroforge_sim/ # C++ SimCore (Bullet physics)
│
├── build/ # Compiled artifacts (ignored in git)
│
├── python/
│ └── aeroforge/
│ ├── core/
│ │ └── simcore_loader.py
│ ├── envs/
│ │ └── drone_nav_env.py
│ └── scripts/
│ └── train_hover_ppo.py
│
└── tests/
└── python/
└── test_drone_nav_env.py

yaml
Copier le code

---

# ⚙️ Installation

### **1. Clone the repo**
```bash
git clone https://github.com/<your-username>/AeroForge.git
cd AeroForge
2. Configure & build the project
bash
Copier le code
cmake -S . -B build
cmake --build build -j$(nproc)
3. Install Python dependencies
bash
Copier le code
pip install numpy gymnasium stable-baselines3 pytest
4. Verify Python bindings
python
Copier le code
import aeroforge_core
sim = aeroforge_core.SimCore()
sim.initialize()
sim.reset()
print(sim.get_observation())
🤖 Simulation Core (C++)
The physics engine is based on Bullet:

Drone rigid body with 6DOF

Ground plane collision

Integration step exposed to Python

Observations returned as a 13-dimensional vector:

csharp
Copier le code
[pos(3), quat(4), lin_vel(3), ang_vel(3)]
Bindings are generated via pybind11, producing:

swift
Copier le code
build/src/bindings/python/aeroforge_core.so
🧠 Python Environment (DroneNavEnv)
DroneNavEnv is a lightweight Gym-like interface around SimCore.

Observation (8D)
csharp
Copier le code
[z, vz, dz, roll, pitch, p, q, r]
Action (4D)
Normalized in [-1,1]:

csharp
Copier le code
[thrust_cmd, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd]
Reward Function
The reward combines:

🛬 Altitude error penalty

🎚️ Vertical velocity penalty

🎛️ Tilt penalty

🔄 Angular rate penalty

📈 Progress reward (reducing |dz|)

🌟 Hover bonus (when stable)

Termination
Max steps

Crash (z < 0.1)

Out-of-bounds (z > 50)

🧪 Unit Tests
Located in: tests/python/test_drone_nav_env.py

Run all tests:

bash
Copier le code
pytest -q
Run only our env test:

bash
Copier le code
pytest tests/python/test_drone_nav_env.py -q
CTEST:

bash
Copier le code
ctest -R python_drone_nav_env --output-on-failure
🎯 Training PPO (Hover Task)
Training script:

bash
Copier le code
python/aeroforge/scripts/train_hover_ppo.py
Start training:

bash
Copier le code
python3 python/aeroforge/scripts/train_hover_ppo.py
Model will be saved to:

bash
Copier le code
models/hover_ppo/ppo_drone_hover.zip
🚁 Evaluation Logging
Evaluation prints detailed telemetry:

z, vz, dz

roll, pitch

angular rates p, q, r

reward breakdown

distance to target

cumulative reward

Example:

vbnet
Copier le code
[Eval] Episode 3 | Step 87
  • Reward this step     : -0.4998
  • Observation          : z=1.043, dz=0.043, roll=1.42, pitch=-0.48
  • Distance to target   : 0.0431
📘 Development Stages (Complete Roadmap)
AeroForge is developed through clear incremental stages.

Stage 1 — Bullet Physics Integration (DONE)
Drone rigid body, gravity, collisions, 13D state.

Stage 2 — Python Bindings (DONE)
Pybind11 module aeroforge_core.

Stage 3 — Minimal RL Environment (DONE)
DroneNavEnv base class, reset/step API, tests.

Stage 4 — Hover RL Task (DONE)
8D observation, 4D normalized action, shaped reward, PPO training.

Stage 5 — Full 3D Navigation (NEXT)
Multi-axis control, XY motion, 3D target.

Stage 6 — Trajectory Following
Waypoints & path tracking.

Stage 7 — Sensor & Noise Models
IMU noise, barometer drift, domain randomization.

Stage 8 — Classical Control Baselines
PID hover, PID attitude, MPC.

Stage 9 — Real Hardware Integration
ROS2, STM32/RPi controller, EKF state estimation, UART/Wi-Fi link.