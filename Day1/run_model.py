import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # 1. Point to your local model folder
    model_path = os.path.join(os.path.dirname(__file__), "smollm")
    
    print(f"[*] Loading model and tokenizer from local folder:\n    {model_path}\n")

    # 2. Load Tokenizer & Model (local_files_only=True ensures completely offline loading)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32, # CPU friendly
    )

    # 3. Define a chat prompt
    messages = [
        {"role": "system", "content": "You are a helpful, concise AI assistant."},
        {"role": "user", "content": "Explain what a neural network is in two sentences."}
    ]

    # Apply the model's official chat template (adds <|im_start|> and <|im_end|>)
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    print("--- [1] Raw Formatted Prompt (ChatML) ---")
    print(prompt)
    print("------------------------------------------\n")

    # 4. Tokenize prompt into integer token IDs
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]
    print(f"[*] Prompt Token Count: {input_len} tokens")

    # 5. Generate output tokens
    print("[*] Generating response...\n")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt itself)
    new_tokens = outputs[0][input_len:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("--- [2] Generated Assistant Reply ---")
    print(reply.strip())
    print("-------------------------------------")

if __name__ == "__main__":
    main()
