import torch
from Leonet_model import LeoNet
import pybullet as p
import pybullet_data
import time
import numpy as np

# --- CONFIG ---
vocab = list("abcdefghijklmnopqrstuvwxyz ")
seq_len = 8
num_joints = 7
weight_file = "leonet_fullrange_pretrained.pth"

# --- Setup PyBullet ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robotId = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

# --- Load LeoNet ---
model = LeoNet(vocab_size=len(vocab))
model.load_state_dict(torch.load(weight_file, map_location='cpu'))
model.eval()

def tokenize(command):
    command = command.lower()
    tokens = [vocab.index(c) if c in vocab else len(vocab)-1 for c in command][:seq_len]
    tokens += [0] * (seq_len - len(tokens))
    return torch.tensor([tokens], dtype=torch.long)

def move_arm_to_pose(joint_targets):
    for i in range(num_joints):
        p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, joint_targets[i])
    for _ in range(480):  # 2 seconds at 240 Hz for smooth motion
        p.stepSimulation()
        time.sleep(1./240.)

while True:
    cmd = input("Enter command (or 'quit' to exit): ").strip().lower()
    if cmd == "quit":
        break
    input_tokens = tokenize(cmd)
    with torch.no_grad():
        logits, motor_pred = model(input_tokens)
    motor_pred = motor_pred.squeeze().numpy()
    # Extract first 7 values as joint targets for KUKA arm
    joint_targets = motor_pred[:num_joints]
    print(f"Moving arm to: {joint_targets} (command: '{cmd}')")
    move_arm_to_pose(joint_targets)
