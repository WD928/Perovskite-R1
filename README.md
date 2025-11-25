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
├── app/
│   ├── app.py                  # Gradio-based web interface (supports Thinking Process visualization)
│   └── requirements.txt        # Dependencies for the demo application
├── Preprocess/
│   ├── gen_paper_cot.py        # Generates Chain-of-Thought (CoT) data using OpenAI o1
│   └── pdf2json.py             # Parses PDF literature into structured JSON format
├── Validation_Records/
│   ├── check_contamination.py  # Script for data integrity and contamination analysis
│   ├── Raw_Model_Outputs.../   # Inference logs for benchmarks
│   └── Human_Expert...         # Expert evaluation records
├── Process_Example.ipynb       # End-to-end tutorial (Preprocessing -> Training -> Inference)
└── README.md            
```

## 🚀 Interactive Demo

We provide a **Gradio-based web interface** that visualizes the model's reasoning process (Chain-of-Thought) separate from the final answer.

### Setup & Run
1. Install dependencies:
   ```bash
   pip install -r app/requirements.txt
   ```
2. Run the application (specify your model path):
   ```bash
   python app/app.py --model_path /path/to/your/model
   ```

## 🛠 Data Processing Pipeline

The `Preprocess/` folder contains the core scripts used to construct the domain-specific dataset:

* **`pdf2json.py`**: Extracts raw text and metadata from scientific PDFs, converting them into a structured JSON format suitable for training.
* **`gen_paper_cot.py`**: A distillation script that utilizes the **OpenAI o1** model to generate high-quality Chain-of-Thought (CoT) reasoning paths based on the raw text.
