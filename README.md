# leonet_robotic

# LeoNet Robotics

A brain-inspired transformer pipeline for mapping natural language commands to continuous, full-range robot arm motions—ready for simulation or real-world robotics research.

---

## 🚀 Overview

**LeoNet Robotics** brings together advanced AI and robotics:
- **Dual-head LeoNet transformer:** Outputs both cognitive (language) and motor (action) signals
- **Expansion–contraction layers:** Bio-inspired architecture for richer representations
- **Trains and infers with easy-to-edit `.jsonl` datasets**
- **PyBullet integration:** Run and demo LeoNet in simulation

---

## 📂 Files

| File                           | Purpose                                                  |
|--------------------------------|----------------------------------------------------------|
| `Leonet_model.py`              | LeoNet model definition (PyTorch)                        |
| `trainer.py`                   | Script to train LeoNet on the dataset                    |
| `leonet_fullrange_dataset.jsonl`| Example dataset with full-range joint positions          |
| `fullrange_inference.py`       | Inference/demo script—type a command, watch the robot!   |
| `README.md`                    | This file (edit to fit your needs)                       |
| `requirements.txt`             | Python dependencies                                      |

---

## 🛠️ Installation

1. **Clone this repository**
    ```bash
    git clone https://github.com/YOURUSERNAME/leonet_robotics.git
    cd leonet_robotics
    ```

2. **Create and activate a Python virtual environment**
    ```bash
    python -m venv leonet-env
    leonet-env\Scripts\activate          # On Windows
    # OR
    source leonet-env/bin/activate       # On macOS/Linux
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏋️‍♂️ Training

Train LeoNet on the provided full-range motion dataset:

```bash
python trainer.py

Run the DEMO in pybullet simulation
python fullrange_inference.py


👏 Credits
LeoNet model architecture:Silpeshkumar j patel

Robotics/Simulation scripts:Silpeshkumawr j patel

Special thanks to PyBullet and the open-source robotics community!
