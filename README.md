# FinCast: A Foundation Model for Financial Time-Series Forecasting

[![Paper](https://img.shields.io/badge/Paper-CIKM%202025-blue)](paper.pdf)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-orange)]()

This repository contains the official implementation of **FinCast**, introduced in our paper:

> **FinCast: A Foundation Model for Financial Time-Series Forecasting**  
> Zhuohang Zhu, Haodong Chen, Qiang Qu, Vera Chung  
> *CIKM 2025* (Accepted)  
> Arxiv link: https://arxiv.org/abs/2508.19609

FinCast is a **decoder-only transformer** trained on over **20B financial time points** across diverse domains and temporal resolutions.  
Technical Highlights:
- **PQ-Loss**: Joint point + probabilistic forecasting.
- **Mixture-of-Experts (MoE)**: Specialization across domains.

---

## 🔥 Features
- Foundation model for **financial time-series forecasting**, flexible input and output length.
- Strong performance in **zero-shot**, **supervised**, and **few-shot** settings.
- Modular architecture with **MoE** and **quantile-aware loss**.
- Scalable to **1 billion of parameters** with efficient inference.

---

## 📦 Installation

- The model weight can be found on 🤗 https://huggingface.co/Vincent05R/FinCast
- The model code can be found on https://github.com/vincent05r/FinCast-fts
- The corresponding datasets to reproduce the results can be found on https://huggingface.co/datasets/Vincent05R/FinCast-Paper-test

Run the env_setup.sh first then run the dep_install.sh. We use conda as the venv management.

## 📊 Experiments

- run the corresponding scripts in the scripts directory to reproduce the results in the paper. The result summary can be generate using the result summary notebook in the notebook directory.


## 📈 Inference

FinCast supports **training-free inference** — simply download a checkpoint and start forecasting.  

✨ **Key Inference Features**  
- ⚡ **Plug-and-Play**: *No training required*. Just load a checkpoint and run inference.  
- 📏 **Flexible Context & Horizon**: Choose **any input length** and **any forecast length**.  
- 🔥 **Flexible time frequency and asset types!**: Choose **any frequency** and **any financial assets** such as stock, crypto, futures, forex.  
- 🎲 **Probabilistic Forecasting**: Native **quantile outputs** for uncertainty & risk analysis.  
- 🎯 **High Accuracy**: State-of-the-art performance across **financial benchmarks**.  

📘 See the **Inference Notebook** for quick start examples.  

Example 1 Apple stock minute data:

![alt text](pics/example1_APPL.png)

---


Example 2 Ethereum minute data:

![alt text](pics/example2_ETHUSD.png)



## ⚡ Future Updates

- PEFT finetune(LORA/DORA) is done, just need to do some testing  
- Covariate Inference(currently implemented the same code as timesfm)  


## 🙌 Credits

- This repository draws inspiration from [TimesFM (Google Research)](https://github.com/google-research/timesfm), as well as [ST-MoE - Pytorch](https://github.com/lucidrains/st-moe-pytorch).  


## ⚠️ Disclaimer

This repository and the FinCast model are provided **for research and educational purposes only**.  
We make **no guarantees** regarding the accuracy, reliability, or suitability of the forecasts for financial decision-making.  

- This software does **not constitute financial advice**.  
- The authors and contributors **are not responsible** for any financial losses, damages, or other consequences arising from the use of this model or its outputs.  
- Users should evaluate and use the model **at their own risk**.  


python run_inference.py `
  --data "1.052, 1.054, 1.051, 1.055, 1.058, 1.060, 1.059, 1.062, 1.065, 1.063, 1.061, 1.066, 1.068, 1.070, 1.069, 1.072, 1.071, 1.075, 1.078, 1.076, 1.074, 1.079, 1.082, 1.084, 1.083, 1.086, 1.085, 1.089, 1.092, 1.090, 1.088, 1.093" `
  --model_path "weights\fincast_v1.pth" `
  --backend "gpu" `
  --data_frequency "m" `
  --context_len 32 `
  --horizon_len 3 `
  --columns_target "Close"
