import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re 
import argparse

parser = argparse.ArgumentParser(description="Perovskite-R1 Gradio Demo")
parser.add_argument(
    "--model_path",
    type=str,
    required=True, 
    help="Path to the directory containing the pre-trained model and tokenizer."
)
args = parser.parse_args()
MODEL_PATH = args.model_path

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
except Exception as e:
    print(f"error: {e}")
    exit()

def predict(message, history):
    messages = []
    for user_turn, bot_turn in history:
        messages.append({"role": "user", "content": user_turn})
        messages.append({"role": "assistant", "content": bot_turn})
    
    messages.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids, 
        max_new_tokens=8192, 
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.8
    )

    response_ids = outputs[0][input_ids.shape[-1]:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    print("="*50)
    print(f"【Raw Output】:\n{response_text}")
    print("="*50)

    pattern = r'(.*)</think>\s*\n*(.*)' 
    
    match = re.search(pattern, response_text, flags=re.DOTALL)
    
    print(f"【Match Object】: {match}")
    print("="*50)
    
    if match:
        thinking_text = match.group(1).strip() 
        answer_text = match.group(2).strip()  

        formatted_response = f"**Thinking:**\n```\n{thinking_text}\n```\n\n**Answer:**\n{answer_text}"
        
        return formatted_response
    else:
        return response_text.strip()

chatbot_component = gr.Chatbot(
    render_markdown=True, 
    show_label=False,
    container=False
)

demo = gr.ChatInterface(
    fn=predict,
    title="Perovskite-R1",
    description="Enter your question to have a conversation with the LLM.",
    theme="soft",
    examples=[["Hello"], ["Do you know about perovskite?"]],
    chatbot=chatbot_component 
)

if __name__ == "__main__":
    demo.launch(share=True)