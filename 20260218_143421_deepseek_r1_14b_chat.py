"""
DeepSeek-R1-Distill-Qwen-14B (AWQ 4-bit) Chat Script

Requires:
  - conda env: agent
  - CC env var for triton: export CC=$(which x86_64-conda-linux-gnu-gcc)
  - GPU: NVIDIA RTX 3090 (24GB VRAM)

Usage:
  CC=/home/syupoh/anaconda3/envs/agent/bin/x86_64-conda-linux-gnu-gcc \
    conda run -n agent python 20260218_143421_deepseek_r1_14b_chat.py

  # Or with a single prompt:
  CC=/home/syupoh/anaconda3/envs/agent/bin/x86_64-conda-linux-gnu-gcc \
    conda run -n agent python 20260218_143421_deepseek_r1_14b_chat.py --prompt "Explain quantum computing"
"""

import argparse
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


MODEL_NAME = "casperhansen/deepseek-r1-distill-qwen-14b-awq"


def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("Loading AWQ model...")
    model = AutoAWQForCausalLM.from_quantized(MODEL_NAME, fuse_layers=False, trust_remote_code=True)
    print("Model loaded.\n")
    return model, tokenizer


def generate(model, tokenizer, messages, max_new_tokens=1024, temperature=0.7):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True)


def interactive_chat(model, tokenizer):
    messages = []
    print("=" * 60)
    print("DeepSeek-R1-Distill-Qwen-14B (AWQ 4-bit)")
    print("Type 'quit' to exit, 'clear' to reset conversation.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Bye!")
            break
        if user_input.lower() == "clear":
            messages.clear()
            print("[Conversation cleared]")
            continue

        messages.append({"role": "user", "content": user_input})
        print("\nAssistant: ", end="", flush=True)

        response = generate(model, tokenizer, messages)
        print(response)

        messages.append({"role": "assistant", "content": response})


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-R1-14B AWQ Chat")
    parser.add_argument("--prompt", type=str, help="Single prompt (non-interactive mode)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max new tokens (default: 1024)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature (default: 0.7)")
    args = parser.parse_args()

    model, tokenizer = load_model()

    if args.prompt:
        messages = [{"role": "user", "content": args.prompt}]
        response = generate(model, tokenizer, messages, args.max_tokens, args.temperature)
        print(response)
    else:
        interactive_chat(model, tokenizer)


if __name__ == "__main__":
    main()
