
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path

MODEL_PATH = "franka_emika_panda/scene.xml"
JOINT_PATH = Path("outputs/panda_joint_trajectory.csv")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)

joint_history = np.loadtxt(
    JOINT_PATH, delimiter=",", skiprows=1, ndmin=2
)

joint_names = [f"joint{i}" for i in range(1, 8)]

qpos_ids = [
    model.jnt_qposadr[model.joint(name).id]
    for name in joint_names
]

body_names = [
    "link0", "link1", "link2", "link3",
    "link4", "link5", "link6", "link7", "hand"
]

body_ids = [model.body(name).id for name in body_names]

frames = []

for q in joint_history:
    data.qpos[qpos_ids] = q
    mujoco.mj_forward(model, data)

    points = np.array([
        data.xpos[body_id].copy()
        for body_id in body_ids
    ])

    frames.append(points)

frames = np.asarray(frames)

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")

robot_line, = ax.plot(
    [], [], [], "o-", linewidth=3, markersize=5
)

trail_line, = ax.plot(
    [], [], [], "--", linewidth=1.5
)

ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.8, 0.8)
ax.set_zlim(0.0, 1.2)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Human Demonstration Driven Panda Robot")

ax.view_init(elev=25, azim=135)

def update(frame):
    points = frames[frame]

    robot_line.set_data(
        points[:, 0], points[:, 1]
    )
    robot_line.set_3d_properties(points[:, 2])

    trail = frames[:frame + 1, -1, :]

    trail_line.set_data(
        trail[:, 0], trail[:, 1]
    )
    trail_line.set_3d_properties(trail[:, 2])

    return robot_line, trail_line

animation = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=50,
    blit=False
)

output = Path("outputs/panda_animation.mp4")

try:
    animation.save(
        output,
        writer=FFMpegWriter(fps=20),
        dpi=120
    )
    print("Animation saved:", output)

except Exception as exc:
    print("MP4 export unavailable:", exc)

    gif_output = Path("outputs/panda_animation.gif")
    animation.save(gif_output, writer="pillow", fps=20)
    print("GIF saved:", gif_output)

plt.show()
