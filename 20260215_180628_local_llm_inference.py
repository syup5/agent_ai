#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
Local LLM 추론(Inference) — 사전학습 vs 지시조정 모델 비교
=============================================================================
사전학습된 LLM은 '다음 토큰 예측'만 수행합니다.
그런데 어떻게 질문에 답하고, 코드를 작성하고, 추론을 수행할 수 있을까요?

비밀은 **지시조정(Instruction Tuning)**에 있습니다:
  - SFT (Supervised Fine-Tuning): 질문-답변 쌍으로 추가 학습
  - RLHF (Reinforcement Learning from Human Feedback): 인간 선호도로 최적화

이 코드는 두 모델을 동일한 질문에 대해 비교합니다:
  1. GPT-2 (사전학습만) → 텍스트 이어쓰기만 가능
  2. Phi-3.5-mini-instruct (지시조정) → 질문 답변, 추론 수행

실행: conda activate agent && python 20260215_180628_local_llm_inference.py
=============================================================================
"""

import time
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 테스트 프롬프트 ──
# 동일한 질문을 두 모델에게 던져 차이를 확인
TEST_PROMPTS = [
    {
        "category": "사실 질문 (Factual Q&A)",
        "question": "What is the capital of France?",
        "base_prompt": "Q: What is the capital of France?\nA:",
    },
    {
        "category": "수학 추론 (Math Reasoning)",
        "question": "A farmer has 15 chickens. 3 flew away and he bought 7 more. How many chickens does the farmer have now?",
        "base_prompt": "Q: A farmer has 15 chickens. 3 flew away and he bought 7 more. How many chickens does the farmer have now?\nA:",
    },
    {
        "category": "번역 (Translation)",
        "question": "Translate the following English sentence to Korean: 'The weather is beautiful today.'",
        "base_prompt": "Translate English to Korean:\nEnglish: The weather is beautiful today.\nKorean:",
    },
    {
        "category": "코드 생성 (Code Generation)",
        "question": "Write a Python function that checks if a given number is prime. Include a docstring.",
        "base_prompt": "# Python function to check if a number is prime\ndef is_prime(n):",
    },
]


# =============================================================================
# Part A: 사전학습 모델 (GPT-2) — 다음 토큰 예측만 수행
# =============================================================================
def run_base_model():
    """
    GPT-2는 인터넷 텍스트로 '다음 토큰 예측'만 학습한 모델입니다.
    질문을 해도 '이 텍스트 다음에 올 법한 내용'을 생성할 뿐,
    질문에 '답변'하도록 학습된 적이 없습니다.
    """
    print("\n" + "=" * 70)
    print("  Part A: 사전학습 모델 (GPT-2) — 다음 토큰 예측")
    print("=" * 70)

    # ── 1단계: 모델 로드 ──
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    print(f"  모델: GPT-2 (117M 파라미터)")
    print(f"  특징: 사전학습만 수행 (지시조정 없음)")

    results = []
    for item in TEST_PROMPTS:
        prompt = item["base_prompt"]
        print(f"\n{'─' * 60}")
        print(f"  [{item['category']}]")
        print(f"  프롬프트:")
        for line in prompt.split('\n'):
            print(f"    {line}")

        # ── 2단계: 토큰화 ──
        inputs = tokenizer(prompt, return_tensors="pt")
        input_len = inputs['input_ids'].shape[1]

        # ── 3단계: 생성 (Greedy Decoding) ──
        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - start

        # ── 4단계: 디코딩 (생성된 토큰 → 텍스트) ──
        generated = tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print(f"  생성된 텍스트:")
        for line in generated[:300].split('\n'):
            print(f"    > {line}")
        print(f"  (토큰 수: {output_ids.shape[1] - input_len}, 시간: {elapsed:.2f}초)")

        results.append({
            "category": item["category"],
            "prompt": prompt,
            "response": generated.strip()[:500],
            "full_text": full_text[:600],
            "tokens_generated": int(output_ids.shape[1] - input_len),
            "time": round(elapsed, 2),
        })

    return results


# =============================================================================
# Part B: 지시조정 모델 (Instruction-Tuned) — 추론 수행
# =============================================================================
def run_instruct_model(model_name="microsoft/Phi-3.5-mini-instruct"):
    """
    Instruction-Tuned 모델은 사전학습 후 추가로:
      1. SFT: (질문, 답변) 쌍 수십만 개로 미세조정
      2. RLHF: 인간이 '좋은 답변'과 '나쁜 답변'을 평가 → 강화학습

    이 과정을 통해 모델은:
      - 질문의 의도를 파악하고
      - 적절한 형식으로 답변하고
      - 단계별 추론을 수행하는
    '행동 패턴'을 학습합니다.
    """
    print("\n" + "=" * 70)
    print(f"  Part B: 지시조정 모델 ({model_name.split('/')[-1]})")
    print("=" * 70)

    # ── 1단계: 모델 로드 (FP16으로 GPU에 로드) ──
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,   # 메모리 절약을 위해 반정밀도
        device_map="auto",           # GPU 자동 매핑
    )
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  모델: {model_name}")
    print(f"  파라미터: {param_count:,} ({param_count/1e9:.1f}B)")
    print(f"  디바이스: {next(model.parameters()).device}")
    print(f"  정밀도: float16")

    results = []
    for item in TEST_PROMPTS:
        print(f"\n{'─' * 60}")
        print(f"  [{item['category']}]")
        print(f"  질문: {item['question']}")

        # ── 2단계: Chat Template 적용 (★ 핵심 차이점) ──
        # 지시조정 모델은 특정 형식의 프롬프트를 기대합니다.
        # apply_chat_template()이 이 형식 변환을 자동으로 처리합니다.
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": item["question"]},
        ]

        # Chat template은 모델마다 다릅니다:
        # Phi-3: <|system|>\n...<|end|>\n<|user|>\n...<|end|>\n<|assistant|>\n
        # Llama: [INST] <<SYS>>\n...\n<</SYS>>\n\n...[/INST]
        # Qwen:  <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        print(f"\n  [Chat Template 적용 결과]")
        for line in input_text.strip().split('\n'):
            print(f"    {line}")

        # ── 3단계: 토큰화 후 GPU로 전송 ──
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_len = inputs['input_ids'].shape[1]

        # ── 4단계: 생성 (Sampling 사용) ──
        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,          # 확률적 샘플링
                temperature=0.7,         # 적당한 다양성
                top_p=0.9,               # 누적 확률 90% 내에서 샘플링
                repetition_penalty=1.1,  # 반복 방지
            )
        elapsed = time.time() - start

        # ── 5단계: 응답 디코딩 (입력 부분 제외) ──
        response = tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )

        print(f"\n  응답:")
        for line in response[:500].split('\n'):
            print(f"    {line}")
        print(f"  (토큰 수: {output_ids.shape[1] - input_len}, 시간: {elapsed:.2f}초)")

        results.append({
            "category": item["category"],
            "question": item["question"],
            "chat_template_applied": input_text,
            "response": response.strip()[:800],
            "tokens_generated": int(output_ids.shape[1] - input_len),
            "time": round(elapsed, 2),
        })

    return results


# =============================================================================
# Part C: 결과 비교 및 저장
# =============================================================================
def compare_and_save(base_results, instruct_results, instruct_model_name):
    """두 모델의 결과를 나란히 비교하고 JSON으로 저장"""
    print("\n" + "=" * 70)
    print("  Part C: 결과 비교")
    print("=" * 70)

    comparisons = []
    for base, inst in zip(base_results, instruct_results):
        print(f"\n{'━' * 60}")
        print(f"  {base['category']}")
        print(f"  ┌─ GPT-2 (사전학습):")
        for line in base['response'][:150].split('\n')[:3]:
            print(f"  │  {line}")
        print(f"  ├─ {instruct_model_name.split('/')[-1]} (지시조정):")
        for line in inst['response'][:150].split('\n')[:3]:
            print(f"  │  {line}")
        print(f"  └─ 시간: GPT-2 {base['time']}s vs Instruct {inst['time']}s")

        comparisons.append({
            "category": base["category"],
            "question": inst["question"],
            "base_model": {
                "name": "GPT-2 (117M)",
                "type": "사전학습 (Pre-trained)",
                "prompt": base["prompt"],
                "response": base["response"],
                "time": base["time"],
            },
            "instruct_model": {
                "name": instruct_model_name.split('/')[-1],
                "type": "지시조정 (Instruction-Tuned)",
                "chat_template": inst.get("chat_template_applied", ""),
                "response": inst["response"],
                "time": inst["time"],
            },
        })

    output = {
        "base_model": "GPT-2",
        "instruct_model": instruct_model_name,
        "test_prompts": TEST_PROMPTS,
        "comparisons": comparisons,
    }
    path = os.path.join(OUTPUT_DIR, "inference_comparison.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[저장됨] {path}")

    return output


def main():
    INSTRUCT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    print("=" * 70)
    print("  Local LLM 추론 — 사전학습 vs 지시조정 모델 비교")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  PyTorch: {torch.__version__}")
    print("=" * 70)

    base_results = run_base_model()
    instruct_results = run_instruct_model(INSTRUCT_MODEL)
    compare_and_save(base_results, instruct_results, INSTRUCT_MODEL)

    print(f"\n완료! 결과: {OUTPUT_DIR}/inference_comparison.json")

# =============================================================================
# 메인 실행
# =============================================================================
if __name__ == "__main__":
    main()
