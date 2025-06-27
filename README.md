# ADAWIFI

ADAWIFI, Collaborative WiFi Sensing for Cross-Environment Adaptation


## Overview

ADAWIFI is a deep learning framework for Wi-Fi–based gesture and activity recognition that explicitly addresses cross-environment deployment challenges. By leveraging multiple Wi-Fi devices (IoT sensors), ADAWIFI collaboratively encodes, aggregates, and adapts sensing data across heterogeneous and dynamic environments with minimal labeled samples.

**Key Innovations:**

1. **Deployment‑Independent Collective Sensing Architecture**:  A permutation‑invariant network that encodes per‑link Doppler Frequency Spectrum (DFS) embeddings via LSTM, aggregates them through a transformer encoder without positional bias, and flexibly accommodates varying numbers and arrangements of devices.
2. **Progressive Tuning with Virtual Domains**:  Constructs intermediate “virtual” domains by mixing sensor streams from source and target environments. Jointly optimizes classification and cross‑domain consistency losses to smoothly transfer knowledge with only 1–10 labeled target samples.
3. **Contribution‑Aware Sensor Reweighting**:  Learns per‑sensor contribution weights in a two‑stage adaptation—first estimating each link’s quality to down‑weight noisy or low‑quality signals, then fine‑tuning the core model—thereby enhancing robustness to signal noise and packet loss.

## Repository Structure

```
adawifi/                  # Root directory
├── adawifi_data_utils.py  # Loading, caching, transforms for CSI (.npy) and DFS (.mat)
├── adawifi_model.py       # Collective sensing network: encoders, aggregator, classifier
├── adawifi_experiments.py # Training, adaptation pipelines, TensorBoard logging
└── README.md              # Project documentation
```

## Installation

```bash
git clone https://github.com/ZhengNaiYu/adawifi.git
cd adawifi
python3 -m venv venv     
source venv/bin/activate
```

## Training and Usage

ADAWIFI supports automatic training and cross-environment adaptation using the `cross_env_experiments` function in `adawifi_experiments.py`.

### Step 1: Run Cross-Environment Experiments

```bash
python adawifi_experiments.py
```

This will:

* Train models (`MixRx_Transformer`, `Client_Transformer`, `Client_Mean`) on two source environments (rooms 0, 1, or 2)
* Evaluate on a target environment with few labeled samples (1-shot, 10-shot, 20-shot)
* Apply three adaptation strategies:

  * **Fine-tuning** on labeled target data
  * **Co-training** with both source and few target samples
  * **Virtual-Domain Adaptation** (for MixRx\_Transformer only)

### Step 2: Customize or Extend

You can modify:

* `model_classes` to try new architectures
* `env_configs` to choose different room-user adaptation targets
* `n_shots_options` to test different few-shot levels
* Add `adv` to `settings` for testing robustness under adversarial conditions

### Output

Results are saved in `cross_room_results.csv`, including per-model accuracy before and after adaptation, and performance under various settings.

## Citation

If you use ADAWIFI, please cite:

```bibtex
@article{zheng2024adawifi,
  title={AdaWiFi, Collaborative WiFi Sensing for Cross-Environment Adaptation},
  author={Zheng, Naiyu and Li, Yuanchun and Jiang, Shiqi and Li, Yuanzhe and Yao, Rongchun and Dong, Chuchu and Chen, Ting and Yang, Yubo and Yin, Zhimeng and Liu, Yunxin},
  journal={IEEE Transactions on Mobile Computing},
  year={2024},
  publisher={IEEE}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*For questions or contributions, please open an issue or contact the authors.*
