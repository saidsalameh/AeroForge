import re
import pathlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# =========================================================
# 1. Locate project root and the waypoint log file
# =========================================================
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]            # .../AeroForge
LOG_FILE = ROOT / "tests" / "logs" / "train_waypoint.txt"
PLOT_DIR = ROOT / "tests" / "plot"

PLOT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. Parse log into structured episodes
# =========================================================
episodes = []
current_epi = []
num_pos_lines = 0
num_episode_headers = 0

if not LOG_FILE.exists():
    raise FileNotFoundError(f"Log file not found: {LOG_FILE}")

with open(LOG_FILE, "r") as f:
    for line in f:
        s = line.strip()

        # -------------------------------------------------
        # Detect start of evaluation episode
        # Examples this should match:
        #   [Eval] Episode 1/5 | Step 3
        #   [Eval] Episode 2
        # -------------------------------------------------
        m_ep = re.search(r'\[Eval\]\s*Episode\s+(\d+)', s)
        if m_ep:
            num_episode_headers += 1
            # Save previous episode if it has data
            if current_epi:
                episodes.append(current_epi)
                current_epi = []
            continue

        # -------------------------------------------------
        # Try multiple patterns for position logs
        # You can adapt these to your exact log format if needed.
        # -------------------------------------------------
        m = None

        # Pattern 1: "pos (x, y, z): 1.0, 2.0, 3.0"
        m = re.search(
            r'pos\s*\(x,\s*y,\s*z\)\s*:\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)',
            s
        )

        # Pattern 2: "pos: x=1.0, y=2.0, z=3.0"
        if m is None:
            m = re.search(
                r'pos\s*:\s*x\s*=\s*([-\d.eE]+)\s*,\s*y\s*=\s*([-\d.eE]+)\s*,\s*z\s*=\s*([-\d.eE]+)',
                s
            )

        # Pattern 3: "pos x=1.0 y=2.0 z=3.0"
        if m is None:
            m = re.search(
                r'pos\s+x\s*=\s*([-\d.eE]+)\s*y\s*=\s*([-\d.eE]+)\s*z\s*=\s*([-\d.eE]+)',
                s
            )

        if m:
            x, y, z = map(float, m.groups())
            current_epi.append((x, y, z))
            num_pos_lines += 1

# Save last episode if present
if current_epi:
    episodes.append(current_epi)

# If we saw positions but no episode headers, treat everything as a single episode
if num_episode_headers == 0 and num_pos_lines > 0 and len(episodes) == 1:
    print("⚠️ No [Eval] Episode headers found; treating all positions as a single episode.")

print(f"📌 Parsed episodes: {len(episodes)}")
print("Lengths:", [len(e) for e in episodes])
print(f"📌 Position lines matched: {num_pos_lines}")


# =========================================================
# 3. Set waypoint target (same one used in training)
# =========================================================
TARGET_POS = [4.0, 4.0, 2.0]   # example target from Stage 6


# =========================================================
# 4. Plot individual episodes
# =========================================================
for i, epi in enumerate(episodes, start=1):
    if not epi:
        continue

    xs = [p[0] for p in epi]
    ys = [p[1] for p in epi]
    zs = [p[2] for p in epi]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot trajectory
    ax.plot(xs, ys, zs, label=f"Episode {i}", linewidth=2)

    # Start point
    x0, y0, z0 = epi[0]
    ax.scatter([x0], [y0], [z0], s=60, marker="o")
    ax.text(x0, y0, z0, "  start", fontsize=9)

    # Target point
    tx, ty, tz = TARGET_POS
    ax.scatter([tx], [ty], [tz], s=80, marker="^")
    ax.text(tx, ty, tz, "  target", fontsize=9)

    # Direction arrows every ~10 steps
    n_points = len(epi)
    stride = max(n_points // 10, 1)

    for k in range(0, n_points - 1, stride):
        x0, y0, z0 = epi[k]
        x1, y1, z1 = epi[k + 1]
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0

        ax.quiver(
            x0, y0, z0,
            dx, dy, dz,
            length=1.0,
            normalize=True,
        )

    # Labels
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Drone waypoint trajectory – Episode {i}")
    ax.legend()

    plt.tight_layout()

    out_file = PLOT_DIR / f"episode_{i}.png"
    fig.savefig(out_file)
    plt.close(fig)

    print(f"📌 Saved plot: {out_file}")
