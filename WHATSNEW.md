# WHATSNEW — agent_ai 변경 이력

---

## v0.1.2 — 2026-02-19 11:49:43

**LLM Agent 아키텍처 연구 레포트 작성**

- `RESEARCH_NOTE/20260219_113254_llm_agent_architecture.md` 작성
  - LLM 기반 Agent 아키텍처: ReAct, Tool Use, Multi-Agent 설계 패턴 분석
  - 약 4000단어 분량의 학술 레포트 (한국어, 기술 용어 영문 병기)
  - 핵심 아키텍처 패턴 4종 분석: ReAct, Tool Use/Function Calling, Plan-and-Execute, Multi-Agent
  - 주요 프레임워크 4종 비교: LangChain/LangGraph, LlamaIndex, AutoGen, CrewAI
  - 메모리 아키텍처 (Short-term/Long-term/Reflective/Hierarchical) 및 RAG 통합 분석
  - 11편의 검증된 논문 인용 (WebSearch/WebFetch로 모든 논문 직접 접속 검증)

---

## v0.1.1 — 2026-02-19 11:20:59

**PhD Dissertation Agent 논문 인용 검증 규칙 추가**

- 글로벌 CLAUDE.md에 phd-dissertation agent 전용 규칙 섹션 추가
- 논문 인용 시 WebFetch/WebSearch로 실제 존재 여부를 직접 접속하여 검증하는 규칙
- 검증 실패 시 인용 제거 또는 대체, 검증 불가한 인용 포함 금지
- 검증 완료된 논문에 접근 가능한 링크(DOI, arXiv 등) 포함 의무화

---

## v0.1.0 — 2026-02-18 15:17:22

**초기 프로젝트 구성**

- 프로젝트 규칙 설정 (CLAUDE.md)
  - 작업 완료 시 로그 저장, 버전 업데이트, README/WHATSNEW 관리 규칙
  - 파일 생성 시 타임스탬프 네이밍 규칙
  - 레포트 작성 규칙 (RESEARCH_NOTE/)
  - Python 실행 결과물 저장 규칙 (results/)
- Git/GitHub 연동 설정
  - Stop hook을 통한 자동 git push
- 디렉토리 구조 생성
  - `log_modification/`, `RESEARCH_NOTE/images/`, `results/images/`
- README.md 작성 (프로젝트 설명, 코드별 설명, 사용법)
- 기존 코드 포함:
  - 토큰화/임베딩/어텐션/텍스트 생성 학습 코드 (01~04)
  - LLM 추론 비교 (GPT-2 vs Qwen2.5)
  - DeepSeek R1 14B AWQ 대화형 채팅
  - Universal Model Loader (FP16/AWQ/BnB4/BnB8 통합 로더)
