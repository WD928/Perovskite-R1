# Perovskite-R1: A Domain-Specialized LLM for Intelligent Discovery of Precursor Additives

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/24xx.xxxxx) 
[![Hugging Face Datasets](https://img.shields.io/badge/🤗%20Hugging%20Face-Datasets-yellow.svg)](https://huggingface.co/datasets/JH976/Perovskite-R1)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue.svg)](https://huggingface.co/JH976/Perovskite-R1)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](./LICENSE)

This is the official repository for the paper **"Perovskite-R1: A Domain-Specialized LLM for Intelligent Discovery of Precursor Additives and Experimental Design"**.

Perovskite-R1 is a large language model fine-tuned on **QwQ-32B**, specifically designed to assist materials scientists in perovskite synthesis planning, precursor selection, and experimental optimization.

---

## 🔗 Quick Links

| Resource | Description | Link |
| :--- | :--- | :--- |
| **Paper** | The full manuscript on arXiv | [Read Paper](https://arxiv.org/abs/2507.16307) |
| **Model** | Perovskite-R1 model weights | [Hugging Face](https://huggingface.co/JH976/Perovskite-R1) |
| **Datasets** | Training set & Task-specific Benchmark | [Hugging Face](https://huggingface.co/datasets/JH976/Perovskite-R1) |
| **Validation** | **Raw outputs, expert reviews & logs** | [Go to Folder](./Validation_Records) |

---

## 📂 Repository Structure

```text
Perovskite-R1/
.
├── app/                       
│   ├── app.py                  
│   └── requirements.txt        
├── Preprocess/                 
│   ├── gen_paper_cot.py      
│   └── pdf2json.py            
├── Validation_Records/   
│   ├── ...          
│   └── ...
├── Process_Example.ipynb      
└── README.md                  
