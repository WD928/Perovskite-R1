# Validation Records & Evaluation Data

This directory contains the detailed validation records, raw experimental logs, and human expert evaluation results for **Perovskite-R1**. These files serve to substantiate the claims made in the manuscript regarding model performance, training data quality, and expert-level proficiency.

## Directory Structure

```text
Validation_Records/
├── Raw_Model_Outputs_and_Scores/      # Raw outputs and scores for all models on the benchmark
├── Human_Expert_Comparison_QwQ_vs_Perovskite-R1.json  # Side-by-side human expert evaluation (N=30)
└── training_set_evaluation.json       # Quality and relevance assessment of the AI-generated training dataset
└── check_contamination.py              # Script to detect data leakage between training and benchmark sets
```

## Detailed Descriptions

1. Raw Model Outputs and Scores
```text
Path: Raw_Model_Outputs_and_Scores/

This folder contains the raw inference logs for Perovskite-R1 and other baseline models (e.g., QwQ-32B, GPT-4o, Llama-3) evaluated on the domain-specific benchmark.

Content: Each JSON file represents one model's performance.

Data Fields: Includes the input question, the model's generated response, and the per-question score (metrics).

Purpose: To provide transparency regarding the quantitative results reported in the paper.
```
2. Human Expert Comparison (Side-by-Side)
```text
File: Human_Expert_Comparison_QwQ_vs_Perovskite-R1.json

This file records the blind side-by-side evaluation conducted by domain experts to verify the "expert-level proficiency" of Perovskite-R1 compared to the base model (QwQ-32B).

Methodology: Experts reviewed 30 randomly selected complex queries (involving synthesis design, defect control, etc.) and selected the better response based on scientific accuracy and logic.

Key Metrics: Better choice.

Purpose: To qualitatively verify the reasoning enhancement achieved through domain-specific fine-tuning.
```
3. Training Set Quality Assessment
```text
File: training_set_evaluation.json

This file addresses the concern regarding the reliability of the AI-generated training data (synthesized from OpenAI o1).

Content: Evaluation of the logical consistency, factual accuracy, and question-answer relevance for a sampled subset of the training set.

Purpose: To demonstrate that the synthetic training data is of high quality and chemically valid, ensuring that the model does not learn from hallucinations or noise.
```
4. Data Contamination Analysis (Sanity Check)
```text
File: check_contamination.py

This script performs a rigorous overlap analysis to ensure the integrity of the evaluation results. It detects potential data leakage between the Training Set and the Evaluation Benchmark.

Methodology: The script utilizes three distinct metrics to identify contamination:
1. Exact Match: Checks for verbatim string inclusion.
2. N-gram Overlap (N=13): Follows the GPT-4 technical report standard to detect partial memorization.
3. Semantic Similarity: Uses 'bge-large-en-v1.5' embeddings (Threshold > 0.88) to detect paraphrased leakage.

Usage: 
$ python check_contamination.py

Output: Generates a console summary and a detailed report (contamination_report_detailed.txt) listing any flagged samples.

Purpose: To guarantee that the reported performance metrics reflect the model's generalization capability rather than memorization of the test set.
```
