# Transformer-Based Autonomous Racing Agent

Transformer-based reinforcement learning project for autonomous racing in Assetto Corsa. This repository combines live simulator control, telemetry collection, human demonstration preprocessing, behavioral cloning, Soft Actor-Critic baselines, and a transformer-based SAC variant for learning racing policies with temporal context.

Repository: [virtual457/transformer-based-autonomous-racing-agent](https://github.com/virtual457/transformer-based-autonomous-racing-agent)

## Overview

This project trains an agent to drive in a real Assetto Corsa session rather than a toy racing environment. The training stack interacts with the simulator through the Assetto Corsa plugin layer, collects telemetry and replay data, builds offline datasets, and trains continuous-control policies for steering, throttle, and brake.

The repo currently includes:

- A live control and telemetry pipeline for Assetto Corsa.
- Behavioral cloning tools for bootstrapping from human driving data.
- MLP-based SAC baselines.
- Transformer-based SAC experiments for sequence-aware control.
- Evaluation scripts, reports, and supporting project documentation.

## What Makes This Project Interesting

- Real simulator integration: the agent interacts with Assetto Corsa rather than a simplified benchmark.
- End-to-end pipeline: collection, preprocessing, imitation learning, RL training, and evaluation live in one repo.
- Temporal modeling: the transformer agent uses a rolling observation window to reason about braking zones, corner entry, and recovery.
- Mixed training strategy: the project supports both human-data initialization and online reinforcement learning.

## Core Components

### 1. Simulator and control stack

- `gym/` contains the local training environment, telemetry parsing, reward shaping, controllers, and SAC implementations.
- `assetto_corsa_gym/` contains the integrated Assetto Corsa plugin and related support code used to interface with the simulator.
- `tests/` contains utility scripts for lifecycle checks, sensor watching, control inspection, and setup validation.

### 2. Learning pipelines

- `gym/sac/` contains the MLP SAC implementation and training entry points.
- `gym/transformer_sac/` contains the transformer-based SAC implementation.
- `AICLONE/` contains the behavioral cloning and offline data preparation pipeline.
- `trained_models/` is reserved for local model artifacts and is intentionally ignored by Git.

### 3. Data and experiment outputs

- `collectDataAI/` contains agent-side collection scripts.
- `human_data/` contains human driving collection and preprocessing scripts.
- `checkpoints/`, `outputs/`, `collected_data/`, and related directories support training runs, logging, and experiment workflows.
- `results/` stores evaluation outputs and supporting analysis artifacts.

## High-Level Architecture

The training flow is organized around a live interaction loop with Assetto Corsa:

1. Launch or connect to the simulator.
2. Read telemetry, sensor, and control-state observations.
3. Build the current observation or observation window.
4. Predict continuous actions with the active policy.
5. Apply steering, throttle, and brake through the control interface.
6. Log rollouts, rewards, and episode statistics.
7. Update the policy from replay data or offline datasets.

The transformer path extends the standard SAC setup by replacing a flat encoder with a temporal encoder over recent observations, which is useful for track approach behavior and smoother high-speed decision making.

## Repository Layout

```text
.
|- AICLONE/                  Behavioral cloning and offline preprocessing
|- Docs/                     Reports and proposal documents
|- assetto_corsa_gym/        Integrated Assetto Corsa plugin and helper library
|- checkpoints/              Local-only training checkpoints
|- collectDataAI/            Agent data collection utilities
|- collected_data/           Local-only collected datasets
|- data/                     Local-only processed datasets and references
|- eval/                     Evaluation scripts
|- gym/                      RL environment, SAC, transformer SAC, rewards, telemetry
|- human_data/               Human driving collection and preprocessing
|- outputs/                  Local-only runtime outputs and lap logs
|- results/                  Selected tracked evaluation artifacts
|- tests/                    Validation and environment utility scripts
|- trained_models/           Local-only model weights and buffers
|- report.md                 Project report
|- ROADMAP.md                Project planning notes
|- ISSUES.md                 Working issue tracker
```

## Getting Started

### Prerequisites

- Windows 10/11
- Assetto Corsa with Content Manager
- Python 3.12
- NVIDIA GPU recommended for training
- vJoy for virtual controller input

### Clone

```bash
git clone https://github.com/virtual457/transformer-based-autonomous-racing-agent.git
cd transformer-based-autonomous-racing-agent
```

### Create a local environment

```bash
python -m venv AssetoCorsa
AssetoCorsa\Scripts\activate
pip install -r assetto_corsa_gym/requirements.txt
```

Depending on your training path, you may also need PyTorch with the CUDA build that matches your machine.

### Assetto Corsa setup

At a minimum, you will need to:

- install the Assetto Corsa plugin assets from `assetto_corsa_gym/`
- configure the virtual controller profile
- verify the simulator can expose telemetry and accept control input

The repo includes setup notes in:

- [assetto_corsa_gym/INSTALL.md](assetto_corsa_gym/INSTALL.md)
- [assetto_corsa_gym/INSTALL_Linux.md](assetto_corsa_gym/INSTALL_Linux.md)
- [assetto_corsa_gym/README.md](assetto_corsa_gym/README.md)

## Common Workflows

### Train the MLP SAC agent

```bash
AssetoCorsa\Scripts\python.exe gym\sac\train_sac.py --manage-ac
```

### Train the transformer SAC agent

```bash
AssetoCorsa\Scripts\python.exe gym\transformer_sac\train.py --manage-ac
```

### Prepare human demonstration data

```bash
AssetoCorsa\Scripts\python.exe human_data\preprocess_human.py
```

### Run behavioral cloning preprocessing

```bash
AssetoCorsa\Scripts\python.exe AICLONE\preprocess_parquet.py
```

### Validate environment connectivity

```bash
AssetoCorsa\Scripts\python.exe tests\test_ac_lifecycle.py
```

## Selected Project Areas

### Reward design

The training environment includes custom reward shaping for progress, stability, and racing behavior. This is one of the main levers used to move from simple lane-following toward competitive lap behavior.

### Data bootstrapping

The repo supports collecting and preprocessing human demonstrations so policies can start from a more stable driving prior before RL fine-tuning.

### Transformer policy experiments

The transformer SAC path is aimed at learning better temporal reasoning than a flat observation encoder can offer, especially for braking and high-speed corner sequences.

## Project Status

This is an active research and engineering project. The repo already contains:

- a working simulator interface
- training code for multiple policy variants
- offline data tooling
- tracked evaluation artifacts
- a cleaned Git history with large local artifacts excluded

Near-term work is centered on improving training stability, comparing transformer and non-transformer policies, and tightening evaluation around lap consistency and control quality.

## Documentation

Additional project material is available in:

- [report.md](report.md)
- [ROADMAP.md](ROADMAP.md)
- [ISSUES.md](ISSUES.md)
- [Docs/](Docs)

## Tech Stack

- Python
- PyTorch
- Reinforcement learning with Soft Actor-Critic
- Transformer sequence models
- Assetto Corsa plugin integration
- vJoy-based control
- NumPy, Pandas, SciPy, Parquet tooling

## Resume-Friendly Summary

If you want to describe this project briefly:

> Built an end-to-end autonomous racing system in Assetto Corsa using telemetry collection, behavioral cloning, Soft Actor-Critic, and transformer-based sequence modeling for continuous vehicle control.

## Contact

Chandan Gowda K S

- Email: `chandan.keelara@gmail.com`
- LinkedIn: [linkedin.com/in/chandan-gowda-k-s-765194186](https://www.linkedin.com/in/chandan-gowda-k-s-765194186/)
- Portfolio: [virtual457.github.io](https://virtual457.github.io/)

## Acknowledgments

- Assetto Corsa by Kunos Simulazioni
- `assetto_corsa_gym` and related simulator integration work
- Soft Actor-Critic research by Haarnoja et al.
- Transformer architecture foundations from Vaswani et al.
