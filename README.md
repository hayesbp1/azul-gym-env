# Azul Gym Environment

This repository contains a Python implementation of the board game Azul, designed as a custom environment for Reinforcement Learning using the Gymnasium library.

## Quick Start

**1. Clone the repository:**
```sh
git clone [https://github.com/your_username/azul-gym-env.git](https://github.com/your_username/azul-gym-env.git)
```

**2. Navigate into the directory:**
```sh
cd azul-gym-env
```

**3. Install dependencies:**
```sh
pip install -r requirements.txt
```

**4. Play the game:**
```sh
python play_game.py
```

## Project Structure

* `azul_env.py`: Contains the `AzulEnv` class, which is the core Gymnasium environment for the game.
* `play_game.py`: A simple script that allows a human to play against a random agent in the command line.
* `requirements.txt`: Lists the necessary Python packages (`gymnasium`, `numpy`).
* `.gitignore`: Standard Python gitignore file.

## Goal

The primary purpose of this environment is to serve as a foundation for training a Reinforcement Learning agent to play Azul.
