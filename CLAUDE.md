# Project Rules - agent_ai

## Conda 환경

- 기본 가상환경: `agent`
- 실행: `conda activate agent`

## 버전 관리

- 현재 버전: **v0.1.0**
- 작업을 완료할 때마다 버전을 업데이트한다 (PATCH 단위 증가, 기능 추가 시 MINOR, 대규모 변경 시 MAJOR).
- 버전 업데이트 시 이 파일의 "현재 버전"도 함께 수정한다.

## 프롬프트 입력 후 작업 완료 시 필수 절차

작업이 완료되면 아래 순서대로 반드시 수행한다:

1. **수정 로그 저장**: `log_modification/` 디렉토리에 MD 파일로 저장.
   - 파일명: `YYYYMMDD_HHMMSS_log.md` (시스템 시간 호출)
   - 내용: 입력된 프롬프트 원문, 수행한 작업 요약, 변경된 파일 목록, 버전 변경 사항
2. **버전 업데이트**: 이 파일(`CLAUDE.md`)의 "현재 버전"을 업데이트.
3. **README.md 업데이트**: 프로젝트 버전, 각 코드 파일에 대한 설명, 코드 사용법(usage)을 반영.
4. **WHATSNEW.md 업데이트**: 새로워진 내용 또는 수정 내용을 추가. 반드시 시스템에서 호출한 작성 날짜와 시간을 포함.
5. **Git 커밋**: 변경된 모든 파일을 `git add` 후 의미있는 커밋 메시지로 커밋.
   - 커밋은 반드시 사용자가 명시적으로 요청하지 않아도 작업 완료 시 자동 수행.
   - Stop hook이 자동으로 `git push`를 수행하므로 push는 하지 않는다.

## 파일 생성 규칙

- 모든 새 파일의 파일명은 `YYYYMMDD_HHMMSS_<설명>.확장자` 형식으로 시작한다.
- 타임스탬프는 반드시 시스템에서 호출한다 (`date '+%Y%m%d_%H%M%S'`). 추정하지 않는다.
- 예외: CLAUDE.md, README.md, WHATSNEW.md, .gitignore, .env, requirements.txt, pyproject.toml 등 설정 파일

## "레포트를 작성하라" 프롬프트 처리

- `./RESEARCH_NOTE/` 폴더에 MD 파일(필요 시 HTML)로 작성.
- 파일명: `YYYYMMDD_HHMMSS_<주제>.md` (시스템 시간 호출)
- 시각화 자료: `./RESEARCH_NOTE/images/` 에 저장.

## Python 코드 실행 결과물 저장

- 시각화 자료 (그래프, 차트 등): `./results/images/` 디렉토리에 저장.
- 숫자/텍스트 결과물 (JSON, CSV 등): `./results/` 디렉토리에 저장.

## 디렉토리 구조

```
agent_ai/
├── CLAUDE.md              # 프로젝트 규칙
├── README.md              # 프로젝트 설명, 버전, 사용법
├── WHATSNEW.md             # 버전별 변경 이력
├── .gitignore
├── log_modification/      # 작업 수정 로그
├── RESEARCH_NOTE/         # 연구 노트 / 레포트
│   └── images/            # 레포트용 시각화 자료
├── results/               # 코드 실행 결과물
│   ├── images/            # 시각화 결과물
│   └── *.json, *.csv      # 숫자 결과물
├── output_images/         # (기존) 출력 이미지
├── output_results/        # (기존) 출력 결과
└── *.py                   # 소스 코드
```

## Git / GitHub

- 원격 저장소: `https://github.com/syup5/agent_ai.git`
- 브랜치: `main`
- Stop hook이 자동으로 push를 수행한다.
