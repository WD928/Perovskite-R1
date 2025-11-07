import os
from openai import OpenAI
import json
import re

model = model
api = api

start_index = 0

client = OpenAI(
    api_key = api,
    base_url = base_url,
)

with open(f"user/dataset_perovskite/paper/paper_split.json", 'r', encoding='utf-8') as file:
    data = json.load(file)

cot_set = []

for idx, item in enumerate(data):
    if idx < start_index:  
        # i += 1
        continue

    excerpt = item['content']

    print("The Number:", idx, "\n", flush=True)

    prompt_user = f"""You are an expert in the field of perovskite. Your task is to generate chain of thought for the construction of dataset.
"""
    prompt_system = """Read and think carefully about the fragment. Generate the suitable output.
"""
    try:
        completion = client.chat.completions.create(
            model = model,
            messages = [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user},
            ],
        )
        output = completion.choices[0].message.content
        print(output,"\n", flush=True)

    except Exception as e:
        output = "Unknown error"
        print(f"Unexpected error at index {idx}: {str(e)}")

    cot_set.append({"id": idx, "cot": output})
    # i += 1

with open(f"user/dataset_perovskite/new/paper_split_cot_new.json", 'w', encoding='utf-8') as file:
    json.dump(cot_set, file, ensure_ascii=False, indent=2)

print("\ndone")
