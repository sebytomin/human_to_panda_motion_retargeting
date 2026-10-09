# Human-to-Panda Motion Retargeting

### Vision-Based Robotic Manipulation from Personally Collected Demonstrations

## Overview

This research aims to study the use of a light method of transferring human manipulation demonstrations to a simulation of Franka Emika Panda robotic arm.

A total of four videos involving manipulations were recorded by the researcher with the help of a smartphone. These movements of objects were captured with the help of OpenCV and projected into the working space of the 7-DOF Panda manipulator.

The approach adopts a manually programmed retargeting pipeline instead of a pretrained Vision-Language-Action policy.

## Technical Pipeline

**Video from Smartphone Camera → Object Detection Using HSV → 2D Trajectories Extraction → Mapping to Workspace Coordinates → Inverse Kinematics Using Jacobian → Joint Configurations for Franka Panda Robot → 3D Visualisation**

### 1. Data Collection

- Four personal recordings of manipulation demonstrations.
- Processing 1,861 video frames.
- Three demonstrations checked visually for consistent tracking.
- Tracking of the blue object feature based on HSV colour segmentation.

### 2. Trajectory Extraction

The tracking of the object is done using OpenCV. The trajectory coordinates are extracted as normalised image coordinates.

These trajectories are saved as CSV files for further use.

### 3. Human-to-Robot Retargeting

Mapping 2D trajectories into a 3D space of a small robot workspace.

Using damped least squares Jacobian inverse kinematics controller to compute joint configurations for Franka Panda robot.

### 4. Simulation and Visualisation

MuJoCo 3.3.0 Panda model is used for robot kinematics.

Generated joint configurations are visualised using Matplotlib.

Currently, the kinematic evaluation is done using direct joint positioning instead of actuation-based manipulation.

## Experimental Results

| Metric | Result |
|---|---|
| Processed demonstration videos | 4 |
| Visually verified tracked demonstrations | 3 |
| Evaluated trajectory targets | 120 |
| Mean end-effector position error | 1.13 mm |
| Maximum end-effector position error | 1.95 mm |
| Robot model | Franka Emika Panda |
| Simulation framework | MuJoCo 3.3.0 |

The reported positional errors measure inverse kinematics tracking of the mapped target positions. They do not represent physical grasp success, real-world positional accuracy, or generalisation to unseen tasks.

## Reproduction

Requirements: Python 3.11, NumPy, OpenCV, Matplotlib, MuJoCo 3.3.0.

Install dependencies:

```bash
pip install numpy opencv-python matplotlib mujoco==3.3.0
```

Download the `franka_emika_panda` model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) and place it in the project root.

Run:

```bash
python extract_trajectory.py
python panda_trajectory.py
python animate_panda.py
```

Place the recorded MP4 demonstrations in `videos/` before running extraction.

## Limitations

- Monocular video does not provide direct metric depth measurements.
- The workspace mapping uses an assumed scale and fixed height.
- Grasp detection and gripper actuation are not implemented.
- Joint trajectories are generated kinematically rather than executed through a closed-loop physics controller.
- No VLA policy was trained or fine-tuned.
- The system has not been evaluated for autonomous pick-and-place success.

## Future Improvements

The future research topics include 3D hand-pose estimation, automatic grasp event detection, manipulation with actuators, task success evaluation, and comparison with imitation learning/VLA policies.

## Contribution

This paper’s main contribution is the complete proof-of-concept, with low computation cost, of generating possible robot joint positions from manually acquired data through computer vision and robotic inverse kinematics.