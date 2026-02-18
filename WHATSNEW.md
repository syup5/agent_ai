# WHATSNEW — agent_ai 변경 이력

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
