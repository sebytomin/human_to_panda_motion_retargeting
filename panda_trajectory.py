
import csv
from pathlib import Path

import mujoco
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = Path("franka_emika_panda/scene.xml")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Select a demonstration with visually verified tracking.
CSV_FILES = sorted(OUTPUT_DIR.glob("*_trajectory.csv"))

if not CSV_FILES:
    raise FileNotFoundError(
        "No trajectory CSV files found in outputs/"
    )

CSV_PATH = CSV_FILES[0]
print("Using demonstration:", CSV_PATH)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)

mujoco.mj_forward(model, data)

hand_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "hand"
)

if hand_id < 0:
    raise RuntimeError("Panda hand body not found")

joint_ids = [
    mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}"
    )
    for i in range(1, 8)
]

qpos_ids = np.array([
    model.jnt_qposadr[j] for j in joint_ids
])

dof_ids = np.array([
    model.jnt_dofadr[j] for j in joint_ids
])

lower = np.array([
    model.jnt_range[j, 0] for j in joint_ids
])

upper = np.array([
    model.jnt_range[j, 1] for j in joint_ids
])

# Read recorded video trajectory.
trajectory = []

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        try:
            x = float(row["x_norm"])
            y = float(row["y_norm"])
        except (ValueError, TypeError):
            continue

        if np.isfinite(x) and np.isfinite(y):
            trajectory.append([x, y])

trajectory = np.asarray(trajectory, dtype=float)

if len(trajectory) < 5:
    raise RuntimeError("Insufficient valid trajectory points")

# Downsample for faster testing.
trajectory = trajectory[::5]

# Map observed image displacement into a small
# robot workspace displacement relative to the
# Panda's initial hand position.
mujoco.mj_forward(model, data)
start_pos = data.xpos[hand_id].copy()

image_start = trajectory[0].copy()

workspace_scale = 0.20  # metres per normalised image unit

targets = []

for x, y in trajectory:
    dx = (x - image_start[0]) * workspace_scale
    dy = -(y - image_start[1]) * workspace_scale

    target = start_pos + np.array([dx, dy, 0.0])
    targets.append(target)

targets = np.asarray(targets)

jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))

actual_positions = []
joint_history = []
errors = []

damping = 0.02
gain = 0.6

for target in targets:

    for iteration in range(100):
        mujoco.mj_forward(model, data)

        current = data.xpos[hand_id].copy()
        error = target - current

        if np.linalg.norm(error) < 0.002:
            break

        jacp[:] = 0
        jacr[:] = 0

        mujoco.mj_jacBody(
            model, data, jacp, jacr, hand_id
        )

        J = jacp[:, dof_ids]

        # Damped least-squares inverse kinematics.
        dq = J.T @ np.linalg.solve(
            J @ J.T + damping**2 * np.eye(3),
            gain * error
        )

        dq = np.clip(dq, -0.05, 0.05)

        q = data.qpos[qpos_ids].copy()
        q = np.clip(q + dq, lower, upper)

        data.qpos[qpos_ids] = q

    mujoco.mj_forward(model, data)

    actual = data.xpos[hand_id].copy()

    actual_positions.append(actual)
    joint_history.append(data.qpos[qpos_ids].copy())
    errors.append(np.linalg.norm(target - actual))

actual_positions = np.asarray(actual_positions)
joint_history = np.asarray(joint_history)
errors = np.asarray(errors)

np.savetxt(
    OUTPUT_DIR / "panda_joint_trajectory.csv",
    joint_history,
    delimiter=",",
    header="joint1,joint2,joint3,joint4,joint5,joint6,joint7",
    comments=""
)

print("\n=== ROBOT TRAJECTORY RESULTS ===")
print("Targets:", len(targets))
print("Mean position error (m):", errors.mean())
print("Maximum position error (m):", errors.max())

# Plot target and achieved paths.
plt.figure(figsize=(8, 6))

plt.plot(
    targets[:, 0],
    targets[:, 1],
    "b--",
    label="Target trajectory"
)

plt.plot(
    actual_positions[:, 0],
    actual_positions[:, 1],
    "r-",
    label="Panda hand trajectory"
)

plt.xlabel("Robot X position (m)")
plt.ylabel("Robot Y position (m)")
plt.title("Human Demonstration to Panda Motion Retargeting")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "panda_trajectory_comparison.png",
    dpi=200
)

plt.show()

print("\nRobot trajectory generation complete.")
