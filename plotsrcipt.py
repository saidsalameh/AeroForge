import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------
# Parse log file into a list of episodes, each = list of (x,y,z)
# ---------------------------------------------------------
episodes = []
current_epi = []

with open("test1.txt") as f:
    for line in f:
        s = line.strip()

        # Detect start of new eval episode
        m_ep = re.search(r'\[Eval\] Episode (\d+)/(\d+)', s)
        if m_ep and current_epi:
            episodes.append(current_epi)
            current_epi = []

        # Extract position line: "pos (x, y, z): ..."
        m = re.search(
            r'pos \(x, y, z\)\s*: *([\d\.\-]+), *([\d\.\-]+), *([\d\.\-]+)',
            s
        )
        if m:
            x, y, z = map(float, m.groups())
            current_epi.append((x, y, z))

# Add last episode if not empty
if current_epi:
    episodes.append(current_epi)

print("Episodes:", len(episodes), "lengths:", [len(e) for e in episodes])

# ---------------------------------------------------------
# Target position (same as in DroneNavEnv for Stage 5)
# ---------------------------------------------------------
target_pos = (12.0, 20.0, 5.0)

# ---------------------------------------------------------
# Plot each episode separately
# ---------------------------------------------------------
for i, epi in enumerate(episodes, start=1):
    if not epi:
        continue

    xs = [p[0] for p in epi]
    ys = [p[1] for p in epi]
    zs = [p[2] for p in epi]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Trajectory line
    ax.plot(xs, ys, zs, label=f"Episode {i}")

    # Start point
    x0, y0, z0 = epi[0]
    ax.scatter([x0], [y0], [z0], s=60, marker="o")
    ax.text(x0, y0, z0, " start", fontsize=9)

    # Target point
    tx, ty, tz = target_pos
    ax.scatter([tx], [ty], [tz], s=80, marker="^")
    ax.text(tx, ty, tz, " target", fontsize=9)

    # Direction arrows (every N steps)
    n_points = len(epi)
    step_stride = max(n_points // 10, 1)  # about 10 arrows per episode

    for k in range(0, n_points - 1, step_stride):
        x0, y0, z0 = epi[k]
        x1, y1, z1 = epi[k + 1]
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        # 3D arrow showing local direction of motion
        ax.quiver(x0, y0, z0, dx, dy, dz, length=1.0, normalize=True)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Drone trajectory – Episode {i}")
    ax.legend()

    plt.tight_layout()
    fig.savefig(f"drone_trajectory_ep{i}.png")

# Optional: show the last figure (or all, depending on backend)
plt.show()
