# Reinforcement learning interest group

## Requirements

- Python 3

## Getting started

1. Ensure that you have Python 3 with `python --version`
2. Create a virtual environment with `python -m venv venv`
3. Activate the virtual environment:
    - Linux/MacOS: `. venv/bin/activate`
    - Windows: `. venv/Scripts/activate`
4. Install requirements:
    - If you're on MacOS: `pip install -r requirements/macos.txt`
    - If you have a CUDA compatible GPU: `pip install -r requirements/cuda.txt`
    - If not: `pip install -r requirements/cpu.txt`

## Training and evaluating the network

- Train the network with `python train.py`
- Evaluate the trained network with `python train.py --eval`
