#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
Universal Model Loader — 다양한 LLM을 통합 로드하는 스크립트
=============================================================================
지원 로딩 방식:
  1. HuggingFace 표준 (FP16/BF16)   — AutoModelForCausalLM
  2. AWQ 양자화                      — AutoAWQForCausalLM
  3. BitsAndBytes 4-bit (NF4/FP4)    — BitsAndBytesConfig
  4. BitsAndBytes 8-bit              — BitsAndBytesConfig

환경:
  conda activate agent

사용법:
  # 모델 목록 보기
  python 20260218_144802_universal_model_loader.py --list

  # 모델 로드 + 대화형 채팅
  python 20260218_144802_universal_model_loader.py --model deepseek-r1-14b-awq
  python 20260218_144802_universal_model_loader.py --model qwen2.5-7b
  python 20260218_144802_universal_model_loader.py --model phi-3.5-mini

  # 단일 프롬프트 모드
  python 20260218_144802_universal_model_loader.py --model qwen2.5-7b --prompt "Hello!"

  # 커스텀 HuggingFace 모델 로드
  python 20260218_144802_universal_model_loader.py --hf "meta-llama/Llama-3-8B-Instruct" --method fp16
  python 20260218_144802_universal_model_loader.py --hf "some/model" --method bnb4

GPU: NVIDIA RTX 3090 (24GB VRAM)
=============================================================================
"""

import argparse
import sys
import time
from dataclasses import dataclass, field

import torch


# =============================================================================
# 모델 레지스트리 — 사전 정의된 모델 프로필
# =============================================================================
@dataclass
class ModelProfile:
    """모델 로딩에 필요한 모든 정보를 담는 프로필"""
    name: str                          # 표시용 이름
    hf_id: str                         # HuggingFace 모델 ID
    method: str                        # 로딩 방식: fp16, bf16, awq, bnb4, bnb8
    description: str = ""              # 모델 설명
    trust_remote_code: bool = True     # trust_remote_code 플래그
    fuse_layers: bool = False          # AWQ fuse_layers 옵션
    chat_template: bool = True         # chat template 사용 여부
    system_prompt: str = "You are a helpful assistant."
    extra_kwargs: dict = field(default_factory=dict)  # 추가 인자


# 사전 정의된 모델 목록
MODEL_REGISTRY: dict[str, ModelProfile] = {
    # ── AWQ 양자화 모델 ──
    "deepseek-r1-14b-awq": ModelProfile(
        name="DeepSeek-R1-Distill-Qwen-14B (AWQ 4-bit)",
        hf_id="casperhansen/deepseek-r1-distill-qwen-14b-awq",
        method="awq",
        description="DeepSeek R1 14B AWQ 양자화 — VRAM ~10GB",
    ),
    "deepseek-r1-7b-awq": ModelProfile(
        name="DeepSeek-R1-Distill-Qwen-7B (AWQ 4-bit)",
        hf_id="casperhansen/deepseek-r1-distill-qwen-7b-awq",
        method="awq",
        description="DeepSeek R1 7B AWQ 양자화 — VRAM ~5GB",
    ),
    "llama3-8b-awq": ModelProfile(
        name="Llama-3-8B-Instruct (AWQ 4-bit)",
        hf_id="casperhansen/llama-3-8b-instruct-awq",
        method="awq",
        description="Meta Llama 3 8B AWQ 양자화 — VRAM ~6GB",
    ),
    "qwen2.5-7b-awq": ModelProfile(
        name="Qwen2.5-7B-Instruct (AWQ 4-bit)",
        hf_id="Qwen/Qwen2.5-7B-Instruct-AWQ",
        method="awq",
        description="Qwen 2.5 7B AWQ 양자화 — VRAM ~5GB",
    ),

    # ── FP16 표준 로딩 ──
    "qwen2.5-7b": ModelProfile(
        name="Qwen2.5-7B-Instruct (FP16)",
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        method="fp16",
        description="Qwen 2.5 7B FP16 — VRAM ~15GB",
    ),
    "phi-3.5-mini": ModelProfile(
        name="Phi-3.5-mini-instruct (FP16)",
        hf_id="microsoft/Phi-3.5-mini-instruct",
        method="fp16",
        description="Microsoft Phi-3.5 3.8B FP16 — VRAM ~8GB",
    ),
    "gemma-2-2b": ModelProfile(
        name="Gemma-2-2B-IT (FP16)",
        hf_id="google/gemma-2-2b-it",
        method="fp16",
        description="Google Gemma 2 2B FP16 — VRAM ~5GB",
    ),

    # ── BitsAndBytes 4-bit 양자화 ──
    "llama3-8b-bnb4": ModelProfile(
        name="Llama-3-8B-Instruct (BnB 4-bit)",
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        method="bnb4",
        description="Meta Llama 3.1 8B NF4 양자화 — VRAM ~6GB",
    ),
    "qwen2.5-14b-bnb4": ModelProfile(
        name="Qwen2.5-14B-Instruct (BnB 4-bit)",
        hf_id="Qwen/Qwen2.5-14B-Instruct",
        method="bnb4",
        description="Qwen 2.5 14B NF4 양자화 — VRAM ~10GB",
    ),
    "mistral-7b-bnb4": ModelProfile(
        name="Mistral-7B-Instruct-v0.3 (BnB 4-bit)",
        hf_id="mistralai/Mistral-7B-Instruct-v0.3",
        method="bnb4",
        description="Mistral 7B NF4 양자화 — VRAM ~5GB",
    ),

    # ── BitsAndBytes 8-bit 양자화 ──
    "qwen2.5-7b-bnb8": ModelProfile(
        name="Qwen2.5-7B-Instruct (BnB 8-bit)",
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        method="bnb8",
        description="Qwen 2.5 7B INT8 양자화 — VRAM ~9GB",
    ),

    # ── 소형 모델 (테스트용) ──
    "gpt2": ModelProfile(
        name="GPT-2 (FP32, 117M)",
        hf_id="gpt2",
        method="fp32",
        description="GPT-2 소형 — 사전학습 전용, 채팅 불가",
        chat_template=False,
    ),
}


# =============================================================================
# 모델 로더
# =============================================================================
class UniversalModelLoader:
    """다양한 방식의 LLM을 통합 로드하는 클래스"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.profile = None
        self._load_time = 0.0

    def load(self, profile: ModelProfile) -> None:
        """프로필에 따라 모델과 토크나이저를 로드"""
        self.profile = profile

        print(f"\n{'=' * 60}")
        print(f"  모델: {profile.name}")
        print(f"  HuggingFace ID: {profile.hf_id}")
        print(f"  로딩 방식: {profile.method.upper()}")
        print(f"{'=' * 60}")

        self._print_vram("로드 전")

        start = time.time()
        loader = {
            "fp16": self._load_fp16,
            "bf16": self._load_bf16,
            "fp32": self._load_fp32,
            "awq": self._load_awq,
            "bnb4": self._load_bnb4,
            "bnb8": self._load_bnb8,
        }

        if profile.method not in loader:
            raise ValueError(f"지원하지 않는 로딩 방식: {profile.method}")

        loader[profile.method](profile)
        self._load_time = time.time() - start

        self._print_model_info()
        self._print_vram("로드 후")
        print(f"  로딩 시간: {self._load_time:.1f}초")
        print(f"{'=' * 60}\n")

    # ── 개별 로더 구현 ──

    def _load_tokenizer(self, profile: ModelProfile):
        from transformers import AutoTokenizer
        print("  토크나이저 로딩...")
        tokenizer = AutoTokenizer.from_pretrained(
            profile.hf_id, trust_remote_code=profile.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_fp16(self, profile: ModelProfile):
        from transformers import AutoModelForCausalLM
        self.tokenizer = self._load_tokenizer(profile)
        print("  모델 로딩 (FP16)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            profile.hf_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=profile.trust_remote_code,
            **profile.extra_kwargs,
        )

    def _load_bf16(self, profile: ModelProfile):
        from transformers import AutoModelForCausalLM
        self.tokenizer = self._load_tokenizer(profile)
        print("  모델 로딩 (BF16)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            profile.hf_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=profile.trust_remote_code,
            **profile.extra_kwargs,
        )

    def _load_fp32(self, profile: ModelProfile):
        from transformers import AutoModelForCausalLM
        self.tokenizer = self._load_tokenizer(profile)
        print("  모델 로딩 (FP32)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            profile.hf_id,
            trust_remote_code=profile.trust_remote_code,
            **profile.extra_kwargs,
        )

    def _load_awq(self, profile: ModelProfile):
        from awq import AutoAWQForCausalLM
        self.tokenizer = self._load_tokenizer(profile)
        print("  모델 로딩 (AWQ 4-bit)...")
        self.model = AutoAWQForCausalLM.from_quantized(
            profile.hf_id,
            fuse_layers=profile.fuse_layers,
            trust_remote_code=profile.trust_remote_code,
            **profile.extra_kwargs,
        )

    def _load_bnb4(self, profile: ModelProfile):
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        self.tokenizer = self._load_tokenizer(profile)
        print("  모델 로딩 (BitsAndBytes NF4)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            profile.hf_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=profile.trust_remote_code,
            **profile.extra_kwargs,
        )

    def _load_bnb8(self, profile: ModelProfile):
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        self.tokenizer = self._load_tokenizer(profile)
        print("  모델 로딩 (BitsAndBytes INT8)...")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            profile.hf_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=profile.trust_remote_code,
            **profile.extra_kwargs,
        )

    # ── 정보 출력 ──

    def _print_model_info(self):
        if self.model is None:
            return
        model = self.model
        # AWQ 모델은 .model 속성에 실제 모델이 있음
        if hasattr(model, "model"):
            model = model.model
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  파라미터: {param_count:,} ({param_count / 1e9:.2f}B)")

    @staticmethod
    def _print_vram(label: str):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  VRAM ({label}): {allocated:.1f}GB 사용 / {total:.1f}GB 전체"
                  f"  (예약: {reserved:.1f}GB)")

    # ── 텍스트 생성 ──

    def generate(self, messages: list[dict], max_new_tokens=512,
                 temperature=0.7, top_p=0.9) -> str:
        """채팅 메시지 → 응답 텍스트 생성"""
        if self.profile.chat_template:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            # chat template이 없는 모델 (예: GPT-2)
            text = messages[-1]["content"]

        inputs = self.tokenizer(text, return_tensors="pt")
        device = self._get_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)

        return self.tokenizer.decode(output[0][input_len:], skip_special_tokens=True)

    def _get_device(self):
        """모델이 위치한 디바이스 반환"""
        model = self.model
        if hasattr(model, "model"):
            model = model.model
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")


# =============================================================================
# 대화형 채팅
# =============================================================================
def interactive_chat(loader: UniversalModelLoader):
    """로드된 모델과 대화형 채팅"""
    messages = []
    system_prompt = loader.profile.system_prompt

    print(f"{'─' * 60}")
    print(f"  {loader.profile.name}")
    print(f"  명령어: quit(종료), clear(초기화), system <msg>(시스템 프롬프트 변경)")
    print(f"{'─' * 60}")

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
            print("[대화 초기화]")
            continue
        if user_input.lower().startswith("system "):
            system_prompt = user_input[7:].strip()
            messages.clear()
            print(f"[시스템 프롬프트 변경: {system_prompt}]")
            continue

        # 메시지 구성
        chat_messages = []
        if system_prompt and loader.profile.chat_template:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.extend(messages)
        chat_messages.append({"role": "user", "content": user_input})

        print("\nAssistant: ", end="", flush=True)
        start = time.time()
        response = loader.generate(chat_messages)
        elapsed = time.time() - start

        print(response)
        print(f"  ({elapsed:.1f}초)")

        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": response})


# =============================================================================
# 모델 목록 출력
# =============================================================================
def print_model_list():
    print(f"\n{'=' * 70}")
    print("  사용 가능한 모델 목록")
    print(f"{'=' * 70}")

    # 방식별 그룹핑
    groups: dict[str, list] = {}
    for key, profile in MODEL_REGISTRY.items():
        groups.setdefault(profile.method, []).append((key, profile))

    method_labels = {
        "awq": "AWQ 4-bit 양자화",
        "fp16": "FP16 (반정밀도)",
        "bf16": "BF16 (반정밀도)",
        "fp32": "FP32 (전체 정밀도)",
        "bnb4": "BitsAndBytes 4-bit (NF4)",
        "bnb8": "BitsAndBytes 8-bit (INT8)",
    }

    for method, items in groups.items():
        print(f"\n  ── {method_labels.get(method, method)} ──")
        for key, profile in items:
            print(f"    {key:<25s} {profile.description}")

    print(f"\n  ── 커스텀 모델 ──")
    print(f"    --hf <model_id> --method <fp16|bf16|awq|bnb4|bnb8>")
    print(f"\n{'=' * 70}")


# =============================================================================
# 메인
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Universal Model Loader — 다양한 LLM 통합 로드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="사용 가능한 모델 목록 출력")
    parser.add_argument("--model", "-m", type=str, help="사전 정의된 모델 키 (--list로 확인)")
    parser.add_argument("--hf", type=str, help="커스텀 HuggingFace 모델 ID")
    parser.add_argument("--method", type=str, default="fp16",
                        choices=["fp16", "bf16", "fp32", "awq", "bnb4", "bnb8"],
                        help="로딩 방식 (default: fp16)")
    parser.add_argument("--prompt", "-p", type=str, help="단일 프롬프트 (비대화형)")
    parser.add_argument("--max-tokens", type=int, default=512, help="최대 생성 토큰 수")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--system", type=str, default="You are a helpful assistant.",
                        help="시스템 프롬프트")

    args = parser.parse_args()

    # GPU 정보
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f}GB)")
    else:
        print("  GPU: 사용 불가 (CPU 모드)")

    # 모델 목록
    if args.list:
        print_model_list()
        return

    # 모델 선택
    if not args.model and not args.hf:
        print("  모델을 지정하세요. --list로 목록을 확인할 수 있습니다.")
        parser.print_help()
        sys.exit(1)

    if args.model:
        if args.model not in MODEL_REGISTRY:
            print(f"  알 수 없는 모델: {args.model}")
            print(f"  사용 가능: {', '.join(MODEL_REGISTRY.keys())}")
            sys.exit(1)
        profile = MODEL_REGISTRY[args.model]
    else:
        profile = ModelProfile(
            name=f"Custom: {args.hf}",
            hf_id=args.hf,
            method=args.method,
            description="사용자 지정 모델",
        )

    profile.system_prompt = args.system

    # 로드
    loader = UniversalModelLoader()
    loader.load(profile)

    # 생성
    if args.prompt:
        messages = [{"role": "user", "content": args.prompt}]
        if profile.system_prompt and profile.chat_template:
            messages.insert(0, {"role": "system", "content": profile.system_prompt})
        response = loader.generate(messages, args.max_tokens, args.temperature)
        print(response)
    else:
        interactive_chat(loader)


if __name__ == "__main__":
    main()
