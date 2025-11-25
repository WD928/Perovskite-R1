#!/usr/bin/env python3
# coding: utf-8

"""
Perovskite Benchmark Data Contamination Analysis Script (Detailed Output)
Checks for overlap between the Training Set and the Evaluation Benchmark.

Metrics:
1. Exact Match (Verbatim string inclusion)
2. N-gram Overlap (N=13, following GPT-4 technical report standard)
3. Semantic Similarity (using BGE-Large embeddings)

Output:
- Console Report
- TXT Report with FULL content of contaminated samples.
"""

import os
import json
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ============================================================

TRAIN_FILE = ""  
BENCH_FILE = ""

RESULT_TXT = ""

N_GRAM_SIZE = 13

MODEL_NAME = "BAAI/bge-large-en-v1.5" 
SEMANTIC_THRESHOLD = 0.88 

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================

def load_json(filepath):
    print(f"Loading {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return []

def normalize_text(text):
    if not text:
        return ""
    return " ".join(text.lower().split())

def generate_ngrams(text: str, n: int) -> set:
    words = re.findall(r'\w+', text.lower())
    if len(words) < n:
        return set()
    return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))

# ============================================================

def run_analysis():
    report_lines = []
    
    def log_report(line):
        print(line)
        report_lines.append(line)

    log_report("="*60)
    log_report(" DATA CONTAMINATION ANALYSIS STARTED (DETAILED MODE)")
    log_report("="*60)
    log_report(f"Metric 1: Exact Match")
    log_report(f"Metric 2: {N_GRAM_SIZE}-gram Overlap")
    log_report(f"Metric 3: Semantic Similarity (Model: {MODEL_NAME}, Threshold: {SEMANTIC_THRESHOLD})")
    log_report("-" * 40)

    train_data = load_json(TRAIN_FILE)
    bench_data = load_json(BENCH_FILE)
    
    print(f"Training Samples: {len(train_data)}")
    print(f"Benchmark Samples: {len(bench_data)}")

    if len(train_data) == 0 or len(bench_data) == 0:
        print("Error: Data is empty.")
        return

    print("\nPreprocessing text data...")

    train_full_texts = [] 
    train_semantic_texts = []
    
    for item in train_data:
        inst = normalize_text(item.get('instruction', ''))
        inp = normalize_text(item.get('input', ''))
        out = normalize_text(item.get('output', ''))
        
        full_text = f"{inst} {inp} {out}"
        train_full_texts.append(full_text)
        
        semantic_text = f"{inst} {out}"
        train_semantic_texts.append(semantic_text)

    bench_texts = []
    for item in bench_data:
        q = normalize_text(item.get('question', ''))
        a = normalize_text(item.get('answer', ''))
        combined = f"{q} {a}"
        bench_texts.append(combined)

    # ------------------------------------------------------------
    # Metric 1: Exact Match Analysis
    # ------------------------------------------------------------
    print(f"\n[1/3] Running Exact Match Analysis...")
    exact_match_indices = []
    
    for idx, b_text in enumerate(tqdm(bench_texts, desc="Checking Exact Matches")):
        if len(b_text) < 50: 
            continue
            
        is_leaked = False
        for t_text in train_full_texts:
            if b_text in t_text:
                is_leaked = True
                break
        
        if is_leaked:
            exact_match_indices.append(idx)

    # ------------------------------------------------------------
    # Metric 2: N-gram Overlap Analysis
    # ------------------------------------------------------------
    print(f"\n[2/3] Running {N_GRAM_SIZE}-gram Overlap Analysis...")
    
    print("  - Building Training N-gram Index...")
    train_ngram_index = set()
    for t_text in tqdm(train_full_texts, desc="Indexing Training Data"):
        grams = generate_ngrams(t_text, N_GRAM_SIZE)
        train_ngram_index.update(grams)
    
    print(f"  - Total Unique {N_GRAM_SIZE}-grams in Training: {len(train_ngram_index)}")
    
    ngram_match_indices = []
    for idx, b_text in enumerate(tqdm(bench_texts, desc="Checking Benchmark N-grams")):
        b_grams = generate_ngrams(b_text, N_GRAM_SIZE)
        if len(b_grams) == 0:
            continue
        
        if not b_grams.isdisjoint(train_ngram_index):
            ngram_match_indices.append(idx)

    del train_ngram_index
    del train_full_texts

    # ------------------------------------------------------------
    # Metric 3: Semantic Similarity Analysis (BGE-Large)
    # ------------------------------------------------------------
    print(f"\n[3/3] Running Semantic Similarity Analysis...")
    print(f"  - Loading Model: {MODEL_NAME}")
    
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    print("  - Encoding Benchmark...")
    bench_embeddings = model.encode(bench_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
    
    print("  - Encoding Training Set (Instruction + Output)...")
    train_embeddings = model.encode(train_semantic_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
    
    print("  - Computing Similarity Matrix...")
    sim_matrix = bench_embeddings @ train_embeddings.T
    
    max_sim_scores, _ = torch.max(sim_matrix, dim=1)
    max_sim_scores = max_sim_scores.cpu().numpy()
    
    semantic_match_indices = np.where(max_sim_scores > SEMANTIC_THRESHOLD)[0].tolist()

    # ------------------------------------------------------------

    total_bench = len(bench_data)
    
    log_report("\n" + "="*60)
    log_report("  FINAL CONTAMINATION REPORT (DETAILED)")
    log_report("="*60)
    
    metrics = {
        "Exact Match": exact_match_indices,
        f"{N_GRAM_SIZE}-gram Overlap": ngram_match_indices,
        f"Semantic (> {SEMANTIC_THRESHOLD})": semantic_match_indices
    }

    all_leaked_indices = sorted(list(set(exact_match_indices) | set(ngram_match_indices) | set(semantic_match_indices)))

    for name, indices in metrics.items():
        count = len(indices)
        rate = (count / total_bench) * 100
        log_report(f"Metric: {name}")
        log_report(f"  - Contaminated Samples: {count} / {total_bench}")
        log_report(f"  - Contamination Rate:   {rate:.2f}%")
        log_report("-" * 40)

    if len(all_leaked_indices) > 0:
        log_report("\n" + "="*60)
        log_report("  DETAILED LIST OF CONTAMINATED SAMPLES")
        log_report("="*60)
        
        for idx in all_leaked_indices:
            item = bench_data[idx]

            reasons = []
            if idx in exact_match_indices: reasons.append("Exact Match")
            if idx in ngram_match_indices: reasons.append("N-gram")
            if idx in semantic_match_indices: reasons.append("Semantic")
            
            log_report(f"\n>>> [Index: {idx}] | Reasons: {', '.join(reasons)}")
            log_report(f"Question: {item.get('question', '').strip()}")
            log_report(f"Answer:   {item.get('answer', '').strip()}")
            log_report("-" * 60)

    clean_rate = 100 - (len(all_leaked_indices) / total_bench * 100)
    
    log_report(f"\nSummary:")
    log_report(f"Total Unique Contaminated Samples: {len(all_leaked_indices)}")
    log_report(f"Clean Benchmark Percentage: {clean_rate:.2f}%")
    
    if len(all_leaked_indices) == 0:
        log_report("\n✅ PASSED: No contamination detected based on current thresholds.")
    else:
        log_report("\n⚠️ WARNING: Potential contamination detected.")

    try:
        with open(RESULT_TXT, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        print(f"\n[SUCCESS] Full detailed report has been saved to: {RESULT_TXT}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save report to file: {e}")

if __name__ == "__main__":
    run_analysis()