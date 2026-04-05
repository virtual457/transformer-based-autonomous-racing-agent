[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

<a id="readme-top"></a>

<div align="center">
  <h3 align="center">🏎️ Transformer-Based Autonomous Racing Agent</h3>
  <p align="center">
    An end-to-end autonomous racing system for Assetto Corsa that combines live simulator control, behavioral cloning, Soft Actor-Critic, and transformer-based sequence modeling for continuous vehicle control.
    <br/>
    <a href="https://github.com/virtual457/transformer-based-autonomous-racing-agent"><strong>Explore the docs</strong></a>
    <br/><br/>
    <a href="https://github.com/virtual457/transformer-based-autonomous-racing-agent">View Repository</a>
    |
    <a href="https://github.com/virtual457/transformer-based-autonomous-racing-agent/issues/new?labels=bug">Report Bug</a>
    |
    <a href="https://github.com/virtual457/transformer-based-autonomous-racing-agent/issues/new?labels=enhancement">Request Feature</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#key-features">Key Features</a></li>
    <li><a href="#technical-architecture">Technical Architecture</a></li>
    <li><a href="#model-architecture">Model Architecture</a></li>
    <li><a href="#training-pipeline">Training Pipeline</a></li>
    <li><a href="#behavioral-cloning-pipeline">Behavioral Cloning Pipeline</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#performance-metrics">Performance Metrics</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## About The Project

This project implements a deep reinforcement learning system that learns to drive inside a live Assetto Corsa session rather than a simplified benchmark environment. The agent interacts with the simulator through the Assetto Corsa plugin stack, consumes telemetry and sensor observations at 25 Hz, and outputs continuous steering, throttle, and brake commands through vJoy-based control.

The repository includes two main learning directions:

- A Soft Actor-Critic baseline with an MLP policy and twin Q-networks.
- A transformer-based SAC variant that uses temporal windows for sequence-aware driving decisions.

### Technical Achievements

- Real-time simulator integration with a live Assetto Corsa control loop
- Continuous-control SAC for steering, throttle, and brake
- Transformer-based policy experiments for temporal racing behavior
- Behavioral cloning pipeline for bootstrapping from human driving data
- Reward shaping and replay-buffer strategies for sparse and unstable racing rewards
- End-to-end workflow covering telemetry collection, preprocessing, training, and evaluation

This project demonstrates experience in:

- Deep Reinforcement Learning
- Imitation Learning
- Sequence Modeling with Transformers
- Real-Time Systems Integration
- Applied ML Infrastructure and Experimentation

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Key Features

### Core RL Implementation
- Continuous-action Soft Actor-Critic with policy and twin Q-networks
- Tanh-squashed Gaussian policy for steer, throttle, and brake
- Replay-buffer training with positive/negative reward balancing
- Polyak target-network updates for stable Bellman targets
- Gradient clipping and observation clamping for stability
- Checkpointed training flow for long-running experiments

### Transformer Variant
- Transformer-based temporal encoder for racing observations
- Rolling-window state representation for sequence-aware control
- Separate encoder path for transformer SAC experiments
- Architecture aimed at better braking and corner-entry decisions
- Sequence replay support for transformer training

### Simulator Integration
- Live Assetto Corsa environment rather than a toy simulator
- Telemetry-driven observation pipeline at 25 Hz
- vJoy-based continuous control output
- Assetto Corsa lifecycle helpers for launch, reset, and collection workflows
- Evaluation and environment validation scripts in `tests/`

### Data and Training Workflow
- Human driving collection and preprocessing scripts
- Behavioral cloning preprocessing and warm-start pipeline
- Agent rollout collection utilities
- Evaluation reports and result artifacts for experiment review

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Technical Architecture

```text
+--------------------------------------------------------------------+
|                    LIVE ASSETTO CORSA TRAINING LOOP                |
|                                                                    |
|   Assetto Corsa <-> Plugin / Socket Layer <-> Our Environment      |
|            ^                         |                    |         |
|            |                         v                    v         |
|         vJoy Control         Observation Builder      Reward Logic  |
|                                                      / Replay Push  |
|                                                            |        |
|                                                            v        |
|                                                SAC / Transformer SAC|
|                                                            |        |
|                                                            v        |
|                                                   Checkpoints / Eval|
+--------------------------------------------------------------------+
```

### Component Interactions

1. Assetto Corsa runs the live driving session.
2. The plugin and environment layer read telemetry, sensor, and control state.
3. The environment builds observations for either the MLP or transformer policy.
4. The policy predicts continuous actions.
5. Actions are mapped to vJoy controls and sent back to the simulator.
6. Rewards, episode transitions, and diagnostics are logged.
7. SAC or transformer SAC updates the policy from replayed experience.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Model Architecture

### MLP SAC Baseline

The baseline SAC policy uses a multi-layer perceptron over telemetry-driven observations. The project report describes a 125-dimensional observation space and 3-dimensional action output:

- Observation space: telemetry, ray sensors, out-of-track flag, curvature look-ahead, action history, and previous observations
- Action space: steer, throttle, brake in continuous policy space
- Twin Q-networks for clipped double-Q learning

### Transformer SAC Variant

The transformer variant replaces the flat encoder with a temporal encoder over a rolling observation window. This is intended to help with:

- braking-zone anticipation
- corner approach timing
- smoother high-speed action selection
- better use of recent temporal context than flat stacked features alone

### Design Rationale

- MLP SAC provides a simpler and faster baseline
- Transformer SAC explores whether temporal modeling improves racing behavior
- Behavioral cloning provides a warm-start path from human driving data before online RL fine-tuning

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Training Pipeline

### High-Level Loop

```python
while training:
    collect_rollouts_from_assetto_corsa()
    push_transitions_to_replay_buffer()
    sample_batches_from_buffer()
    update_policy_and_critics()
    save_checkpoints_and_logs()
```

### Training Stages

**1. Collection**
- connect to or launch Assetto Corsa
- drive with the current policy
- record observations, actions, rewards, and termination signals

**2. Replay and learning**
- sample experience batches
- update critics from Bellman targets
- update the actor using entropy-regularized SAC objectives
- refresh target networks with soft updates

**3. Evaluation and iteration**
- checkpoint models
- inspect logs and evaluation reports
- refine rewards, observation design, or initialization strategy

### Why This Pipeline Matters

Racing agents are sensitive to reward shaping, unstable early exploration, and temporal dynamics. This repo is structured to support fast iteration across those three pressure points instead of treating training as a single-script black box.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Behavioral Cloning Pipeline

The repo includes an imitation-learning path through `AICLONE/` and `human_data/` to initialize policies from demonstrations before online RL training.

### Pipeline Goals

- collect human driving traces
- preprocess demonstrations into model-ready datasets
- learn a driving prior before online reinforcement learning
- compare pure RL against demonstration-bootstrapped policies

### Included Components

- `human_data/collect_human.py`
- `human_data/preprocess_human.py`
- `AICLONE/preprocess_parquet.py`
- `AICLONE/pretrain_actor.py`
- `AICLONE/generate_target_speed.py`
- `AICLONE/finetune_on_demo.py`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites

**Hardware**
- NVIDIA GPU recommended for training
- 16 GB RAM recommended
- Windows 10/11

**Software**
- Assetto Corsa
- Content Manager
- Python 3.12
- vJoy

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/virtual457/transformer-based-autonomous-racing-agent.git
cd transformer-based-autonomous-racing-agent

# 2. Create a local environment
python -m venv AssetoCorsa
AssetoCorsa\Scripts\activate

# 3. Install dependencies
pip install -r assetto_corsa_gym/requirements.txt
```

Depending on your setup, you may also need to install the CUDA-compatible PyTorch build for your GPU.

### Simulator Setup

You will need to configure:

- the Assetto Corsa plugin files from `assetto_corsa_gym/`
- the vJoy controller profile
- the simulator environment and telemetry flow

Helpful references:

- [assetto_corsa_gym/README.md](assetto_corsa_gym/README.md)
- [assetto_corsa_gym/INSTALL.md](assetto_corsa_gym/INSTALL.md)
- [assetto_corsa_gym/INSTALL_Linux.md](assetto_corsa_gym/INSTALL_Linux.md)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### Train the MLP SAC agent

```bash
AssetoCorsa\Scripts\python.exe gym\sac\train_sac.py --manage-ac
```

### Train the transformer SAC agent

```bash
AssetoCorsa\Scripts\python.exe gym\transformer_sac\train.py --manage-ac
```

### Prepare human data

```bash
AssetoCorsa\Scripts\python.exe human_data\preprocess_human.py
```

### Run behavioral cloning preprocessing

```bash
AssetoCorsa\Scripts\python.exe AICLONE\preprocess_parquet.py
```

### Validate environment lifecycle

```bash
AssetoCorsa\Scripts\python.exe tests\test_ac_lifecycle.py
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Project Structure

```text
transformer-based-autonomous-racing-agent/
|
|-- gym/                         RL environment, SAC, transformer SAC, rewards
|   |-- sac/                     MLP SAC implementation
|   |-- transformer_sac/         Transformer SAC implementation
|   |-- telemetry/               Telemetry parsing and environment support
|   |-- rewards/                 Reward components and composition
|
|-- AICLONE/                     Behavioral cloning and offline preprocessing
|-- human_data/                  Human driving collection and preprocessing
|-- collectDataAI/               Agent rollout collection utilities
|-- assetto_corsa_gym/           Assetto Corsa plugin and integration layer
|-- eval/                        Evaluation scripts
|-- tests/                       Setup and validation scripts
|-- results/                     Evaluation outputs and reports
|-- report.md                    Technical report
|-- README.md                    This file
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Performance Metrics

### Environment Characteristics

| Metric | Value |
|--------|-------|
| Control frequency | 25 Hz |
| Simulator | Assetto Corsa |
| Control type | Continuous steer, throttle, brake |
| Main learning setup | SAC + transformer SAC experiments |

### Model and Training Highlights

| Area | Detail |
|------|--------|
| Baseline approach | MLP-based Soft Actor-Critic |
| Sequence approach | Transformer-based SAC |
| Demonstration support | Human-data preprocessing + behavioral cloning |
| Integration mode | Live simulator control with telemetry feedback |

### Why These Metrics Matter

This project is less about a leaderboard score and more about solving the engineering problem of training a stable autonomous racing policy in a live simulator loop with realistic control, temporal reasoning, and data bootstrapping.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

### Completed
- [x] Live Assetto Corsa integration
- [x] Continuous-control SAC training pipeline
- [x] Behavioral cloning preprocessing pipeline
- [x] Transformer SAC implementation path
- [x] Evaluation scripts and result tracking
- [x] Project documentation and technical reporting

### In Progress
- [ ] Improve training stability and policy quality
- [ ] Compare MLP and transformer variants more systematically
- [ ] Tighten evaluation around lap consistency and control smoothness

### Planned
- [ ] Stronger ablation studies across reward and architecture choices
- [ ] More polished evaluation summaries and visualizations
- [ ] Broader multi-track or multi-condition experiments

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Built With

### Core ML
* [PyTorch](https://pytorch.org/) - Deep learning and policy training
* [CUDA](https://developer.nvidia.com/cuda-toolkit) - GPU acceleration

### RL and Data Tooling
* [NumPy](https://numpy.org/) - Numerical computation
* [Pandas](https://pandas.pydata.org/) - Data handling
* [SciPy](https://scipy.org/) - Scientific computing utilities

### Simulator and Control
* [Assetto Corsa](https://www.assettocorsa.it/) - Racing simulator
* [vJoy](https://sourceforge.net/projects/vjoystick/) - Virtual controller interface
* [assetto_corsa_gym](https://github.com/dasGringuen/assetto_corsa_gym) - Integration base for simulator communication

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Contributions are welcome. Helpful areas include:

- reward-design experiments
- transformer architecture improvements
- behavioral cloning refinements
- evaluation tooling
- documentation and reproducibility improvements

If you want to contribute:

1. Fork the project
2. Create a feature branch
3. Commit your changes
4. Push the branch
5. Open a pull request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

**Chandan Gowda K S**
- Email: chandan.keelara@gmail.com
- LinkedIn: [Chandan Gowda K S](https://www.linkedin.com/in/chandan-gowda-k-s-765194186/)
- Portfolio: [virtual457.github.io](https://virtual457.github.io/)

**Project Link**: [https://github.com/virtual457/transformer-based-autonomous-racing-agent](https://github.com/virtual457/transformer-based-autonomous-racing-agent)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/virtual457/transformer-based-autonomous-racing-agent.svg?style=for-the-badge
[forks-shield]: https://img.shields.io/github/forks/virtual457/transformer-based-autonomous-racing-agent.svg?style=for-the-badge
[stars-shield]: https://img.shields.io/github/stars/virtual457/transformer-based-autonomous-racing-agent.svg?style=for-the-badge
[issues-shield]: https://img.shields.io/github/issues/virtual457/transformer-based-autonomous-racing-agent.svg?style=for-the-badge
[contributors-url]: https://github.com/virtual457/transformer-based-autonomous-racing-agent/graphs/contributors
[forks-url]: https://github.com/virtual457/transformer-based-autonomous-racing-agent/network/members
[stars-url]: https://github.com/virtual457/transformer-based-autonomous-racing-agent/stargazers
[issues-url]: https://github.com/virtual457/transformer-based-autonomous-racing-agent/issues
