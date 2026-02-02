<div align="center">

# 🌦️ LM-Weather

<p align="center">
  <em><b>Personalized Adapter for Large Meteorology Model on Devices</b></em><br>
  <em>Towards Weather Foundation Models</em>
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=llTroju97T">
    <img src="https://img.shields.io/badge/NeurIPS%202024-Paper-%23FF6B6B?style=for-the-badge&logo=openreview&logoColor=white" alt="NeurIPS 2024 Paper">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-%234ECDC4?style=for-the-badge&logo=opensourceinitiative&logoColor=white" alt="MIT License">
  </a>
  <a href="https://github.com/shengchaochen82/LM-Weather/stargazers">
    <img src="https://img.shields.io/github/stars/username/LM-Weather?style=for-the-badge&color=FFD93D&logo=github&logoColor=white" alt="GitHub Stars">
  </a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,3,30&height=150&section=header&text=Weather%20Foundation%20Models&fontSize=40&fontColor=fff&animation=twinkling" alt="Header">
</p>

</div>

---

## 🚀 What's New

<table>
<tr>
<td width="100%">

| 📅 Date | 🎉 Update |
|---------|-----------|
| **2025.01** | 🚀 ~~Datasets under external review~~ **Datasets Released!** Available on [Google Drive](https://drive.google.com/file/d/19OuWAH_nJEJQ7AnMu6kJLEofhRHGDcuE/view?usp=sharing) |
| **2024.12** | 🔥 Official code implementation released + Full tutorials |
| **2024.09** | 📢 Paper accepted at **NeurIPS 2024**! |

</td>
</tr>
</table>

---

## 📊 Project Status

<div align="center">

| ✅ Completed | 📦 Status |
|:------------:|:---------:|
| Release Code | ![Done](https://img.shields.io/badge/✓-Done-brightgreen) |
| Training Tutorials | ![Done](https://img.shields.io/badge/✓-Done-brightgreen) |
| Dataset License | ![Done](https://img.shields.io/badge/✓-Done-brightgreen) |
| Tutorial Documentation | ![Done](https://img.shields.io/badge/✓-Done-brightgreen) |
| Paper Publication | ![Done](https://img.shields.io/badge/✓-Done-brightgreen) |

</div>

> 💡 **Tip**: Want to explore on different time series domains? Try our framework on your own datasets following the format from [Time-Series-Library](https://github.com/thuml/Time-Series-Library).

---

## 🎯 Overview

<p align="center">
  <img src="assest/image.png" width="85%" alt="LM-Weather Architecture" style="border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
</p>

**LM-Weather** demonstrates that **Pre-trained Language Models (PLMs)** can serve as powerful foundation models for on-device meteorological modeling. Our approach enables:

<div align="center">

| Feature | Description |
|---------|-------------|
| 🎨 **Customized Models** | Tailored models for heterogeneous meteorological data |
| ⚡ **Efficient Knowledge Fusion** | Low-rank transmission for global knowledge integration |
| 🔒 **Privacy-Preserving** | Secure communication between clients and server |
| 🏆 **State-of-the-Art** | Strong performance on forecasting & imputation tasks |

</div>

---

## 📖 Abstract

<p align="justify">

This paper demonstrates that **pre-trained language models (PLMs)** are strong foundation models for on-device meteorological variable modeling. We present **LM-Weather**, a generic approach to taming PLMs that have learned massive sequential knowledge from natural language databases to acquire immediate capabilities for highly customized models on heterogeneous meteorological data devices while maintaining high efficiency.

Our key contributions:

🧠 **Weather Pattern Awareness** — Lightweight personalized adapter infused into PLMs  
🌐 **Global Knowledge Fusion** — Low-rank-based transmission during federated communication  
🔐 **Privacy & Efficiency** — High communication efficiency with privacy guarantees  
📈 **Superior Performance** — Outperforms SOTA across various forecasting & imputation tasks  

Extensive experiments verify that LM-Weather can:
1. ✅ Leverage sequential knowledge from natural language for accurate meteorological prediction
2. ✅ Enable highly customized models under significant data heterogeneity
3. ✅ Generalize under data-limited and out-of-distribution (OOD) scenarios

</p>

---

## 🚦 Quick Start

### 📥 Clone Repository

```bash
git clone https://github.com/username/LM-Weather.git
cd LM-Weather
```

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔧 Model & Data Preparation

<details>
<summary><b>🤖 Foundation Models (Click to expand)</b></summary>

| Model | Source | Download |
|-------|--------|----------|
| **GPT2** | OpenAI / Hugging Face | [Download Weights](https://huggingface.co/openai-community/gpt2/tree/main) |
| **Open-LLAMA** | OpenLM Research | [3B v2](https://huggingface.co/openlm-research/open_llama_3b_v2) |
| **BERT** | Google | [Base Uncased](https://huggingface.co/google-bert/bert-base-uncased) |

</details>

<details>
<summary><b>📚 Libraries (Click to expand)</b></summary>

- **PEFT**: [Hugging Face PEFT](https://github.com/huggingface/peft) — Parameter-Efficient Fine-Tuning
- **Baselines**: [Time-Series-Library](https://github.com/thuml/Time-Series-Library) — Time series models

</details>

---

## ⚙️ Configuration Guide

### 📋 Training Parameters

<div align="center">

| Parameter | Value | Description |
|-----------|-------|-------------|
| `seq_len` | **192** | Input sequence length |
| `pred_len` | **[96,192,336,720]** | Prediction lengths for different horizons |
| `label_len` | **48** | Training label length |
| `batch_size` | **256** | Training batch size |
| `num_clients` | **15** | Number of federated clients |
| `join_ratio` | **0.1** | Client participation ratio per round |
| `local_steps` | **5** | Local training steps |
| `learning_rate` | **0.005** | Learning rate |
| `eval_gap` | **5** | Evaluation frequency |
| `global_rounds` | **50** | Global communication rounds |

</div>

### 🏗️ Model Configuration

```python
base_model = "GPT2"          # Foundation model: GPT2, BERT, or LLAMA
freeze_part = 2              # Number of frozen layers
gpt_layers = 1               # GPT layers to use
lradj = "type1"              # Learning rate adjustment
```

### 🔧 PEFT Settings

```python
is_peft = 1                  # Enable PEFT
peft_method = "lora"         # LoRA adaptation
rank = 32                    # LoRA rank
```

### 📊 Data Configuration

```python
features = "MS"              # Multivariate input, Single output
freq = "h"                   # Hourly frequency
target = "rh"                # Target: relative humidity (customizable)
dataset = "Weather-Tiny"     # Dataset name (customizable)
```

### 🚀 Run Training

```bash
bash scripts/long_term_forecast.sh
```

---

## 📈 Performance Highlights

<div align="center">

| 🏆 Achievement | 📊 Result |
|:--------------:|:---------:|
| **State-of-the-Art** | Outperforms existing methods by large margin |
| **Forecasting** | Multiple time horizons: 96, 192, 336, 720 |
| **Imputation** | Robust missing data recovery |
| **Scalability** | Efficient on-device deployment |

</div>

---

## 📚 Citation

If you find our research helpful, please cite:

```bibtex
@inproceedings{chen2024personalized,
  title={Personalized Adapter for Large Meteorology Model on Devices: Towards Weather Foundation Models},
  author={Shengchao Chen and Guodong Long and Jing Jiang and Chengqi Zhang},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=llTroju97T}
}
```

---

## 🙏 Acknowledgements

> **⚠️ Note**: We are actively reshaping our code. Some changes may occur.

We gratefully acknowledge these excellent open-source frameworks:

<p align="center">
  <a href="https://github.com/thuml/Time-Series-Library">
    <img src="https://img.shields.io/badge/Time--Series--Library-Repository-blue?style=flat-square&logo=github">
  </a>
  <a href="https://github.com/huggingface/peft">
    <img src="https://img.shields.io/badge/HuggingFace%20PEFT-Repository-yellow?style=flat-square&logo=huggingface">
  </a>
  <a href="https://github.com/TsingZ0/PFLlib">
    <img src="https://img.shields.io/badge/PFLlib-Repository-green?style=flat-square&logo=github">
  </a>
</p>

---

## 📜 License

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-4ECDC4?style=for-the-badge&logo=opensourceinitiative&logoColor=white" alt="MIT License">
</p>

<div align="center">
  <p><i>© 2024 Shengchao Chen. All rights reserved.</i></p>
  
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,3,30&height=100&section=footer" alt="Footer">
</div>
