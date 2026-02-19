# agent_ai

**Version: v0.1.1**

LLM 추론(Inference) 학습 및 실험 프로젝트. 다양한 로컬 LLM 모델을 로드하고, 비교하고, 대화형으로 사용하는 코드 모음.

## 환경

- **Conda 환경**: `agent`
- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **주요 패키지**: transformers, autoawq, bitsandbytes, accelerate, torch (CUDA)

## 코드 설명

### 1. `20260215_160921_01_tokenization.py`
토큰화(Tokenization) 기초 학습 코드. 텍스트를 토큰으로 변환하는 과정을 시각적으로 보여줌.

### 2. `20260215_160921_02_embedding.py`
임베딩(Embedding) 기초 학습 코드. 토큰을 벡터로 변환하는 과정을 설명.

### 3. `20260215_160921_03_attention.py`
어텐션(Attention) 메커니즘 학습 코드. Self-Attention의 작동 원리를 시각화.

### 4. `20260215_160921_04_text_generation.py`
텍스트 생성 기초 학습 코드. 다음 토큰 예측 기반 텍스트 생성 과정 설명.

### 5. `20260215_160921_llm_visualization.ipynb`
LLM 내부 동작을 시각화하는 Jupyter 노트북.

### 6. `20260215_180628_local_llm_inference.py`
사전학습 모델(GPT-2)과 지시조정 모델(Qwen2.5-7B-Instruct)을 비교하는 추론 코드.

### 7. `20260215_180628_llm_inference_explained.html`
LLM 추론 과정을 설명하는 HTML 문서.

### 8. `20260218_143421_deepseek_r1_14b_chat.py`
DeepSeek-R1-Distill-Qwen-14B (AWQ 4-bit) 모델을 사용한 대화형 채팅 스크립트.

### 9. `20260218_144802_universal_model_loader.py`
다양한 LLM을 통합 로드하는 유니버설 모델 로더. FP16/BF16, AWQ, BitsAndBytes 4-bit/8-bit 양자화 지원.

## Usage

```bash
# 환경 활성화
conda activate agent

# DeepSeek R1 14B 대화형 채팅
CC=/home/syupoh/anaconda3/envs/agent/bin/x86_64-conda-linux-gnu-gcc \
  python 20260218_143421_deepseek_r1_14b_chat.py

# 단일 프롬프트 모드
CC=/home/syupoh/anaconda3/envs/agent/bin/x86_64-conda-linux-gnu-gcc \
  python 20260218_143421_deepseek_r1_14b_chat.py --prompt "Explain quantum computing"

# Universal Model Loader — 모델 목록 보기
python 20260218_144802_universal_model_loader.py --list

# Universal Model Loader — 모델 로드 + 대화
python 20260218_144802_universal_model_loader.py --model deepseek-r1-14b-awq
python 20260218_144802_universal_model_loader.py --model qwen2.5-7b-awq
python 20260218_144802_universal_model_loader.py --model phi-3.5-mini

# Universal Model Loader — 단일 프롬프트
python 20260218_144802_universal_model_loader.py -m qwen2.5-7b-awq -p "Hello!"

# Universal Model Loader — 커스텀 모델
python 20260218_144802_universal_model_loader.py --hf "some/model-id" --method bnb4

# 사전학습 vs 지시조정 모델 비교
python 20260215_180628_local_llm_inference.py
```

## 디렉토리 구조

```
agent_ai/
├── CLAUDE.md              # 프로젝트 규칙 (Claude Code용)
├── README.md              # 이 파일
├── WHATSNEW.md             # 버전별 변경 이력
├── .gitignore
├── log_modification/      # 작업 수정 로그
├── RESEARCH_NOTE/         # 연구 노트 / 레포트
│   └── images/
├── results/               # 코드 실행 결과물
│   └── images/
├── output_images/         # 출력 이미지 (기존)
├── output_results/        # 출력 결과 (기존)
└── *.py                   # 소스 코드
```
