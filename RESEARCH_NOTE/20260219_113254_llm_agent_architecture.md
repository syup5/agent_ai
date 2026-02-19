# LLM 기반 Agent 아키텍처: ReAct, Tool Use, Multi-Agent 설계 패턴 분석

**작성일**: 2026-02-19
**분류**: 연구 노트 / 아키텍처 분석
**키워드**: LLM Agent, ReAct, Tool Use, Function Calling, Multi-Agent System, LangChain, AutoGen

---

## 1. 서론

### 1.1 LLM Agent의 정의

대규모 언어 모델(Large Language Model, LLM)은 텍스트 생성과 이해 능력에서 괄목할 만한 성과를 보여왔으나, 단독으로는 외부 세계와의 상호작용, 실시간 정보 접근, 복잡한 다단계 문제 해결에 한계를 지닌다. **LLM Agent**란 LLM을 핵심 추론 엔진(reasoning engine)으로 활용하면서, 외부 도구(tool) 호출, 환경과의 상호작용, 메모리 관리, 계획 수립 등의 능력을 갖춘 자율적 시스템을 의미한다. Wang et al. (2023)의 서베이에 따르면, LLM 기반 자율 에이전트는 프로파일 모듈(profile module), 메모리 모듈(memory module), 계획 모듈(planning module), 행동 모듈(action module)로 구성되며, 이들이 유기적으로 결합하여 복잡한 과제를 자율적으로 수행한다 [1].

### 1.2 배경 및 필요성

전통적인 LLM 활용 방식은 단일 프롬프트에 대한 단일 응답 생성(single-turn generation)에 국한되었다. 그러나 실세계 문제는 대개 다음과 같은 특성을 지닌다:

- **다단계 추론**: 하나의 질문에 답하기 위해 여러 번의 정보 검색과 추론이 필요
- **외부 도구 활용**: 계산기, 검색 엔진, 데이터베이스, API 등 외부 자원과의 연동 필수
- **동적 계획 수립**: 초기 계획이 실행 과정에서 수정되어야 하는 상황 빈발
- **장기 기억 관리**: 대화 이력과 학습된 경험을 유지하고 활용

이러한 요구에 부응하여 LLM Agent 아키텍처 연구가 급격히 발전하고 있으며, 2023년 이후 ReAct, Toolformer, Reflexion 등 다양한 패러다임이 제안되었다. Wei et al. (2022)이 Chain-of-Thought(CoT) 프롬프팅을 통해 LLM의 추론 능력을 끌어낸 이후 [2], 이를 행동(action)과 결합하려는 시도가 Agent 연구의 기폭제가 되었다.

### 1.3 레포트의 목적과 구성

본 레포트는 LLM Agent 아키텍처의 핵심 설계 패턴을 체계적으로 분석한다. 먼저 ReAct, Tool Use, Plan-and-Execute, Multi-Agent 등 주요 아키텍처 패턴을 기술적으로 분석하고, 이를 구현하는 대표 프레임워크들을 비교한다. 이어서 메모리 및 상태 관리 전략을 논의하고, 최신 동향과 미래 방향을 전망한다.

---

## 2. 핵심 아키텍처 패턴

### 2.1 ReAct (Reasoning + Acting) 패턴

#### 2.1.1 개요 및 동기

Yao et al. (2023)이 제안한 **ReAct** 프레임워크는 LLM의 추론(reasoning)과 행동(acting)을 교차적으로(interleaved) 수행하는 패러다임이다 [3]. 기존 연구에서 추론 능력(예: Chain-of-Thought)과 행동 생성(예: action plan generation)은 별개의 연구 주제로 다루어져 왔으나, ReAct는 이 둘의 시너지(synergy)를 극대화한다.

#### 2.1.2 작동 메커니즘

ReAct의 핵심은 **Thought-Action-Observation** 루프에 있다:

1. **Thought (사고)**: 현재 상황을 분석하고 다음 행동을 계획하는 추론 단계. LLM이 자연어로 내부 추론 과정을 명시적으로 생성한다.
2. **Action (행동)**: 외부 환경이나 도구와 상호작용하는 단계. 검색 엔진 쿼리, API 호출, 환경 명령 등을 실행한다.
3. **Observation (관찰)**: 행동의 결과를 관찰하고 피드백을 수신하는 단계. 이 결과가 다음 Thought의 입력이 된다.

```
Thought 1: 이 질문에 답하려면 먼저 X에 대한 정보가 필요하다.
Action 1: Search["X에 대한 정보"]
Observation 1: X는 2024년에 발표된 기술로, ...
Thought 2: X에 대한 정보를 얻었으니, 이제 Y와의 관계를 파악해야 한다.
Action 2: Search["X와 Y의 관계"]
Observation 2: X와 Y는 ... 의 관계를 가진다.
Thought 3: 충분한 정보를 수집했으므로 최종 답변을 구성한다.
Action 3: Finish["최종 답변 내용"]
```

#### 2.1.3 실험적 성과

ReAct는 HotPotQA(질의응답), FEVER(사실 검증), ALFWorld(텍스트 기반 게임), WebShop(웹 네비게이션) 등 4개 벤치마크에서 평가되었다. 특히 ALFWorld에서는 1-2 shot ReAct 프롬프팅만으로 10^3 ~ 10^5개의 학습 인스턴스로 훈련된 모방/강화학습 방법 대비 34%의 성공률 향상을 달성하였다 [3]. 이는 ReAct가 적은 예제만으로도 강력한 에이전트 성능을 발휘할 수 있음을 보여준다.

#### 2.1.4 학술적 의의

ReAct의 핵심 기여는 추론 과정의 **해석 가능성(interpretability)**에 있다. Thought 단계를 통해 에이전트의 의사결정 과정이 자연어로 노출되므로, 인간이 에이전트의 판단 근거를 추적하고 진단할 수 있다. 이는 블랙박스 방식의 행동 생성과 대비되는 중요한 장점이며, 후속 연구들(Reflexion, LATS 등)의 기반이 되었다.

### 2.2 Tool Use / Function Calling 메커니즘

#### 2.2.1 Tool Use의 필요성

LLM은 학습 데이터에 기반한 확률적 생성 모델이므로, 정확한 수학 연산, 실시간 정보 검색, 구조화된 데이터 접근 등에서 본질적 한계를 가진다. **Tool Use**(도구 사용)는 이러한 한계를 외부 도구와의 연동을 통해 극복하는 패러다임이다.

#### 2.2.2 Toolformer: 자가 학습 기반 도구 사용

Schick et al. (2023)이 제안한 **Toolformer**는 LLM이 스스로 도구 사용법을 학습하는 최초의 체계적 접근이다 [4]. Toolformer의 핵심 아이디어는 다음과 같다:

- **자기 지도 학습(Self-supervised Learning)**: 소수의 도구 사용 예시만으로 모델이 언제, 어떤 도구를, 어떤 인자로 호출해야 하는지 학습
- **API 호출 삽입**: 학습 데이터의 텍스트 시퀀스 내에 도구 호출 토큰을 삽입하여, 다음 토큰 예측 성능이 향상되는 위치에서 자동으로 도구 사용을 학습
- **다양한 도구 통합**: 계산기, QA 시스템, 검색 엔진 2종, 번역 시스템, 캘린더 등 6종의 도구를 통합

Toolformer는 다양한 다운스트림 태스크에서 zero-shot 성능을 크게 향상시켰으며, 훨씬 큰 모델과 비슷한 수준의 성능을 달성하였다 [4].

#### 2.2.3 Function Calling: 산업 표준의 등장

OpenAI가 2023년 6월에 도입한 **Function Calling** API는 LLM의 도구 사용을 산업적으로 표준화한 이정표이다. 개발자가 함수의 이름, 설명, 파라미터 스키마를 JSON Schema 형식으로 정의하면, 모델이 사용자 입력을 분석하여 적절한 함수 호출과 인자를 JSON 형태로 출력한다. 이후 `functions` 파라미터에서 `tools` 파라미터로의 진화, 그리고 2025년의 Responses API를 통한 에이전틱 루프(agentic loop) 지원까지 빠르게 발전해왔다.

#### 2.2.4 Gorilla: API 호출 특화 모델

Patil et al. (2023)은 **Gorilla**를 통해 LLM의 API 호출 능력을 전문화하는 접근을 제안하였다 [5]. LLaMA 기반 모델을 HuggingFace, TorchHub, TensorHub 등의 11,000개 이상의 API 데이터셋(APIBench)으로 파인튜닝하여, GPT-4를 능가하는 API 호출 정확도를 달성하였다. 특히 문서 검색기(retriever)와 결합할 경우 할루시네이션(hallucination)을 크게 억제하면서도 API 변경에 대한 적응력을 유지하였다.

#### 2.2.5 Model Context Protocol (MCP)

2024년 11월 Anthropic이 발표한 **Model Context Protocol(MCP)**은 LLM과 외부 도구 간의 연동을 위한 개방형 표준 프로토콜이다. MCP는 M개의 LLM과 N개의 도구를 연결하는 MxN 문제를 단일 프로토콜로 해결하고자 한다. JSON-RPC 2.0 기반의 클라이언트-서버 아키텍처를 채택하며, Language Server Protocol(LSP)의 메시지 흐름 패턴을 재활용한다. 서버는 Prompts, Resources, Tools의 세 가지 프리미티브를, 클라이언트는 Roots와 Sampling의 두 가지 프리미티브를 지원한다. 2025년에는 OpenAI도 MCP 지원을 선언하였으며, 2025년 12월에는 Anthropic이 Linux Foundation 산하의 Agentic AI Foundation(AAIF)에 MCP를 기증하여 업계 표준으로서의 위상을 공고히 하였다.

### 2.3 Plan-and-Execute 패턴

#### 2.3.1 개요

Plan-and-Execute 패턴은 **계획 수립(planning)**과 **실행(execution)**을 명시적으로 분리하는 아키텍처이다. ReAct가 추론과 행동을 매 단계마다 교차 수행하는 반면, Plan-and-Execute는 먼저 전체 과제에 대한 실행 계획을 수립한 후, 각 하위 과제(subtask)를 순차적으로 실행한다.

#### 2.3.2 Plan-and-Solve 프롬프팅

Wang et al. (2023)이 제안한 **Plan-and-Solve(PS) Prompting**은 Zero-shot Chain-of-Thought의 한계(계산 오류, 단계 누락, 의미 오해)를 극복하기 위한 방법이다 [6]. PS Prompting은 두 단계로 구성된다:

1. **계획 수립**: 전체 과제를 더 작은 하위 과제들로 분할하는 계획을 수립
2. **순차 실행**: 수립된 계획에 따라 하위 과제들을 순서대로 수행

GPT-3 기반 실험에서 PS Prompting은 모든 데이터셋에서 Zero-shot-CoT를 큰 폭으로 능가하였으며, 8-shot CoT 프롬프팅과 비슷한 수학 추론 성능을 보였다 [6].

#### 2.3.3 HuggingGPT: 태스크 계획과 모델 선택의 분리

Shen et al. (2023)이 제안한 **HuggingGPT**는 Plan-and-Execute 패턴의 대표적 구현이다 [7]. HuggingGPT는 다음 4단계 파이프라인으로 동작한다:

1. **Task Planning**: LLM이 사용자 요청을 파싱하여 태스크 목록을 생성하고, 실행 순서와 자원 의존성을 결정
2. **Model Selection**: Hugging Face에 등록된 전문가 모델(expert model)들의 설명을 기반으로 각 태스크에 적합한 모델을 선택
3. **Task Execution**: 선택된 전문가 모델들이 할당된 태스크를 실행
4. **Response Generation**: LLM이 전문가 모델들의 추론 결과를 통합하여 최종 응답을 생성

이 아키텍처는 LLM을 "컨트롤러(controller)"로, 특화 모델들을 "실행자(executor)"로 분리함으로써, 단일 모델의 한계를 넘어서는 복합 AI 시스템을 구현하였다. NeurIPS 2023에서 발표된 이 연구는 이후 다양한 Agent 프레임워크의 설계에 영향을 미쳤다.

#### 2.3.4 ReAct vs Plan-and-Execute: 비교 분석

| 특성 | ReAct | Plan-and-Execute |
|------|-------|-------------------|
| **계획 방식** | 단계별 동적 계획 | 사전 전체 계획 수립 |
| **유연성** | 높음 (매 단계 재계획 가능) | 중간 (재계획 시 비용 발생) |
| **효율성** | 단계별 LLM 호출로 비용 증가 | 계획 1회 + 실행 N회로 효율적 |
| **복잡한 의존성** | 암묵적 처리 | 명시적 의존성 그래프 |
| **오류 복구** | 자연스러운 재시도 | 전체 재계획 또는 부분 수정 필요 |
| **적용 분야** | 탐색적 과제, QA | 명확한 목표의 복합 과제 |

### 2.4 Multi-Agent 시스템

#### 2.4.1 단일 Agent의 한계와 Multi-Agent의 동기

단일 LLM Agent는 복잡한 과제에서 다음과 같은 한계를 보인다:

- **역할 혼재(role confusion)**: 하나의 에이전트가 기획, 코딩, 검토, 디버깅 등 다수 역할을 동시에 수행할 때 성능 저하
- **컨텍스트 길이 제한**: 긴 대화와 대량의 도구 출력으로 인한 컨텍스트 윈도우 초과
- **자기 수정의 어려움**: 자신의 출력을 스스로 비판적으로 검토하는 능력의 한계

Multi-Agent 시스템은 여러 에이전트에게 전문화된 역할을 부여하고, 에이전트 간 협력과 대화를 통해 이러한 한계를 극복한다.

#### 2.4.2 AutoGen: 다중 에이전트 대화 프레임워크

Microsoft Research의 Wu et al. (2023)이 제안한 **AutoGen**은 다중 에이전트 대화(multi-agent conversation)를 통해 차세대 LLM 애플리케이션을 구현하는 프레임워크이다 [8]. AutoGen의 핵심 설계 원칙은 다음과 같다:

- **커스터마이저블 에이전트**: 각 에이전트는 LLM, 인간 입력, 도구의 조합으로 동작 모드를 설정 가능
- **대화 기반 협력**: 에이전트 간 자연어 대화와 코드를 통한 유연한 상호작용 패턴 정의
- **인간 참여(Human-in-the-loop)**: 필요 시 인간이 대화에 참여하여 에이전트를 지도하거나 검증

AutoGen은 수학, 코딩, 질의응답, 의사결정, 엔터테인먼트 등 다양한 도메인에서 효과를 입증하였다. 2024년에는 AutoGen 0.4로 대규모 리아키텍처를 거치면서 비동기(asynchronous) 메시징 기반의 확장 가능한 아키텍처로 진화하였다.

#### 2.4.3 CrewAI: 역할 기반 다중 에이전트 오케스트레이션

2024년 1월 출시된 **CrewAI**는 역할 기반(role-playing) 다중 에이전트 프레임워크로서, 각 에이전트에게 명확한 역할(role), 목표(goal), 배경 이야기(backstory)를 부여한다. CrewAI의 주요 특징은 다음과 같다:

- **독립적 아키텍처**: LangChain 등 외부 프레임워크에 의존하지 않고 독자적으로 구축
- **다양한 실행 모델**: 순차(sequential), 병렬(parallel), 조건부(conditional) 처리 지원
- **CrewAI Flows**: 엔터프라이즈 환경을 위한 프로덕션 아키텍처

CrewAI는 2025년 기준 100,000건 이상의 일일 에이전트 실행을 처리하며, PwC, IBM, NVIDIA 등 Fortune 500 기업의 60%가 채택한 것으로 보고되고 있다.

#### 2.4.4 Generative Agents: 사회 시뮬레이션

Park et al. (2023)은 **Generative Agents**를 통해 LLM 에이전트의 사회적 시뮬레이션 가능성을 제시하였다 [9]. 25명의 에이전트가 The Sims와 유사한 가상 환경에서 일상 생활을 영위하며, 기억의 저장, 반성(reflection), 그리고 행동 계획이라는 아키텍처를 통해 인간과 유사한 행동을 창발적으로(emergently) 보여주었다. 이 연구는 Agent의 메모리 아키텍처 설계에 중요한 기여를 하였으며, ACM UIST 2023에서 발표되었다.

#### 2.4.5 Multi-Agent 설계 패턴의 분류

Multi-Agent 시스템의 설계 패턴은 크게 네 가지로 분류할 수 있다:

| 패턴 | 설명 | 대표 사례 |
|------|------|----------|
| **협력적 토론 (Cooperative Debate)** | 에이전트들이 동등한 위치에서 토론하여 합의에 도달 | Multi-Agent Debate |
| **계층적 위임 (Hierarchical Delegation)** | 상위 에이전트가 하위 에이전트에게 과제 위임 | AutoGen의 Manager 패턴 |
| **파이프라인 (Pipeline)** | 에이전트들이 순차적으로 과제를 처리하여 전달 | CrewAI의 Sequential Process |
| **경쟁적 생성 (Competitive Generation)** | 여러 에이전트가 독립적으로 해법을 제시하고 최선을 선택 | LLM Debate |

---

## 3. 주요 프레임워크 비교

### 3.1 LangChain / LangGraph

**LangChain**은 2022년 10월 출시 이후 LLM 애플리케이션 개발의 사실상 표준(de facto standard)으로 자리잡았다. 체인(Chain) 기반의 모듈러 아키텍처로 프롬프트 관리, 메모리, 도구 통합, 검색 증강 생성(RAG) 등을 지원한다.

그러나 복잡한 에이전트 워크플로우의 한계를 보완하기 위해 2024년 초 **LangGraph**가 별도 라이브러리로 출시되었다. LangGraph는 에이전트 워크플로우를 **상태 머신(state machine)**으로 모델링한다:

- **Node**: 현재 상태를 입력받아 LLM 호출이나 도구 실행 등의 태스크를 수행하고, 상태를 업데이트하여 반환
- **Edge**: 노드 간 실행 흐름을 정의하는 논리 규칙
- **StateGraph**: 전체 컨텍스트를 유지하는 중앙집중식 상태 관리

LangGraph의 주요 장점은 다음과 같다:
- **내구적 실행(Durable Execution)**: 실패를 통과해도 지속되며, 중단점에서 재개 가능
- **Human-in-the-loop**: 임의 시점에서 에이전트 상태를 검사하고 수정 가능
- **포괄적 메모리**: 진행 중인 추론을 위한 단기 메모리와 세션 간 장기 메모리 모두 지원

### 3.2 LlamaIndex

**LlamaIndex**(구 GPT Index)는 데이터 통합과 RAG에 특화된 프레임워크이다. 2022년 11월 출시 이후 44,000개 이상의 GitHub 스타와 LlamaHub을 통한 300개 이상의 데이터 커넥터를 보유하고 있다.

LlamaIndex의 에이전트 오케스트레이션은 **Workflows** 모듈을 통해 구현되며, 이벤트 기반(event-driven)의 비동기 우선(async-first) 아키텍처를 채택한다. 이는 LangGraph의 명시적 그래프 정의와 대비되는 접근으로, Pythonic한 추상화를 통해 개발자 친화적인 경험을 제공한다.

### 3.3 AutoGen

앞서 2.4.2절에서 기술한 AutoGen은 Microsoft Research가 주도하는 Multi-Agent 대화 프레임워크이다. 2024년의 대규모 리아키텍처(v0.4)를 통해 비동기 메시징 기반으로 전환하였으며, 다음과 같은 강점을 보인다:
- **유연한 대화 패턴**: 1:1, 그룹 채팅, 계층적 위임 등 다양한 대화 토폴로지 지원
- **코드 실행 통합**: Docker 기반 샌드박스에서 안전한 코드 실행
- **인간-에이전트 협력**: 에이전트 대화에 인간이 자연스럽게 참여

### 3.4 프레임워크 비교 요약

| 특성 | LangChain/LangGraph | LlamaIndex | AutoGen | CrewAI |
|------|---------------------|------------|---------|--------|
| **핵심 강점** | 범용 에이전트 오케스트레이션 | 데이터/RAG 특화 | Multi-Agent 대화 | 역할 기반 오케스트레이션 |
| **아키텍처** | 상태 머신 그래프 | 이벤트 기반 Workflows | 대화 기반 메시징 | 역할-태스크 매핑 |
| **학습 곡선** | 높음 (커스텀 구문 필요) | 중간 (Pythonic API) | 중간 | 낮음 (직관적 추상화) |
| **프로덕션 적합성** | 높음 (LangSmith 통합) | 높음 (엔터프라이즈 버전) | 중간-높음 | 높음 (CrewAI Enterprise) |
| **외부 의존성** | 독자 생태계 | 독자 생태계 | 독자 아키텍처 | 독립적 (외부 의존 없음) |
| **커뮤니티 규모** | 최대 | 대형 | 대형 | 급성장 중 |

---

## 4. 메모리 및 상태 관리

### 4.1 메모리의 중요성

LLM Agent에서 메모리는 단순한 대화 이력 저장을 넘어, 에이전트의 학습, 적응, 자기 개선을 가능하게 하는 핵심 구성 요소이다. Park et al. (2023)의 Generative Agents 연구에서 메모리 아키텍처의 정교함이 에이전트 행동의 사실성(believability)에 결정적 영향을 미친다는 것이 실증되었다 [9].

### 4.2 Short-term Memory (단기 기억)

단기 기억은 현재 진행 중인 과제의 컨텍스트를 유지하는 역할을 한다. 기술적으로는 LLM의 **컨텍스트 윈도우(context window)** 내에 유지되는 정보를 의미하며, 다음 요소들을 포함한다:

- **대화 이력(Conversation History)**: 사용자와 에이전트 간 대화의 최근 N턴
- **작업 상태(Working Memory)**: 현재 과제의 중간 결과, 변수, 도구 출력
- **Scratchpad**: ReAct의 Thought-Action-Observation 트레이스

컨텍스트 윈도우 한계를 극복하기 위한 전략으로는 슬라이딩 윈도우(sliding window), 요약 기반 압축(summary-based compression), 토큰 중요도 기반 선택적 보존(selective retention) 등이 활용된다.

### 4.3 Long-term Memory (장기 기억)

장기 기억은 세션 간 지속되는 정보를 관리하며, 에이전트의 경험 축적과 자기 개선을 지원한다. 구현 방식에 따라 다음과 같이 분류된다:

#### 4.3.1 벡터 데이터베이스 기반 외부 메모리

경험, 문서, 과거 상호작용 등을 임베딩 벡터로 변환하여 벡터 데이터베이스(Pinecone, Weaviate, ChromaDB 등)에 저장한다. 검색 시에는 현재 쿼리와의 유사도(similarity)를 기반으로 관련 기억을 검색한다.

#### 4.3.2 반성적 기억 (Reflective Memory)

Shinn et al. (2023)이 제안한 **Reflexion** 프레임워크는 에이전트가 언어적 피드백(verbal feedback)을 통해 자기 자신을 강화학습하는 패러다임이다 [10]. 에이전트는 과제 수행 후 실패 원인을 자연어로 반성하고, 이 반성 내용을 에피소딕 메모리 버퍼에 저장하여 이후 시도에서 활용한다. Reflexion은 가중치 업데이트 없이도 여러 코드 생성 벤치마크에서 최첨단 성능을 달성하였다.

#### 4.3.3 계층적 기억 (Hierarchical Memory)

Park et al. (2023)의 Generative Agents는 세 수준의 기억 계층을 도입하였다 [9]:

1. **관찰 기억(Observation Memory)**: 에이전트가 직접 경험한 사건의 원시 기록
2. **반성 기억(Reflection Memory)**: 다수의 관찰을 종합하여 생성한 고차원 추상화
3. **계획 기억(Plan Memory)**: 미래 행동에 대한 의도와 스케줄

기억 검색 시에는 **최신성(recency)**, **중요도(importance)**, **관련성(relevance)**의 세 가지 점수를 결합한 가중 합산을 사용한다.

### 4.4 RAG (Retrieval-Augmented Generation) 통합

Lewis et al. (2020)이 제안한 **RAG**는 매개변수적 기억(parametric memory, 즉 모델 가중치)과 비매개변수적 기억(non-parametric memory, 즉 외부 문서 인덱스)를 결합하는 프레임워크이다 [11]. LLM Agent의 맥락에서 RAG는 장기 기억의 핵심 구현 기술로 활용된다:

- **지식 기반 연동**: 도메인 특화 문서, 매뉴얼, 데이터베이스를 에이전트의 지식으로 통합
- **할루시네이션 억제**: 외부 근거(evidence)에 기반한 응답 생성으로 사실적 정확성 향상
- **동적 업데이트**: 모델 재학습 없이 새로운 정보를 즉시 반영

최근에는 에이전트가 자체적으로 RAG 파이프라인을 제어하는 **Agentic RAG** 패턴이 등장하였다. 에이전트가 검색 쿼리를 동적으로 생성하고, 검색 결과의 적합성을 판단하며, 필요 시 쿼리를 재구성하는 능동적 검색 전략을 수행한다.

### 4.5 상태 관리 패턴

LLM Agent의 상태(state) 관리는 메모리 관리와 밀접하게 연관되며, 프레임워크별로 다양한 패턴이 존재한다:

- **LangGraph**: TypedDict 또는 Pydantic 모델로 명시적 상태 스키마를 정의하고, 각 노드가 상태를 읽고 업데이트
- **AutoGen**: 대화 이력 자체가 암묵적 상태로 기능하며, 에이전트 간 메시지 교환을 통해 상태 전달
- **CrewAI**: 태스크별 결과물이 다음 태스크의 입력으로 자동 전달되는 파이프라인 상태 관리

---

## 5. 최신 동향 및 미래 방향

### 5.1 2024-2025년 주요 동향

#### 5.1.1 에이전틱 코딩 (Agentic Coding)의 부상

2025년은 "에이전틱 코딩"의 해로 불리고 있다. GitHub Copilot Agent, Cursor, Claude Code, Devin 등 코딩 에이전트가 개발자 생산성을 혁신적으로 향상시키고 있으며, 단순 코드 자동완성을 넘어 프로젝트 전체의 이해, 다중 파일 편집, 테스트 작성, 디버깅까지 수행한다.

#### 5.1.2 추론 시간 확장 (Test-time Scaling)

OpenAI의 o1/o3, DeepSeek R1 등 추론 시간에 더 많은 연산을 투입하여 복잡한 문제를 해결하는 접근이 에이전트 아키텍처에도 영향을 미치고 있다. 에이전트가 각 단계에서 더 깊은 추론을 수행하고, Chain-of-Thought를 자동으로 확장하는 방식이 탐구되고 있다.

#### 5.1.3 멀티모달 에이전트 (Multimodal Agent)

텍스트뿐 아니라 이미지, 비디오, 음성 등 다양한 모달리티를 처리하는 멀티모달 에이전트가 등장하고 있다. GPT-4V, Gemini 등의 멀티모달 LLM을 기반으로, 화면 이해(screen understanding), 물리적 환경 탐색, 다중 감각 정보 통합 등의 능력이 에이전트에 부여되고 있다.

#### 5.1.4 MCP의 업계 표준화

앞서 2.2.5절에서 기술한 MCP는 2025년 월 9,700만 건의 SDK 다운로드, 10,000개 이상의 활성 서버를 기록하며 도구 통합의 사실상 표준으로 자리잡았다. Python, TypeScript, C#, Java SDK가 제공되며, OpenAI, Google 등 주요 LLM 제공자들이 MCP 지원을 선언하였다.

### 5.2 미래 연구 방향

#### 5.2.1 자율적 자기 개선 (Autonomous Self-Improvement)

현재 에이전트의 학습은 대부분 프롬프팅이나 인컨텍스트 학습에 의존하지만, 향후에는 에이전트가 자율적으로 경험을 축적하고, 전략을 최적화하며, 도구 사용 패턴을 개선하는 방향으로 발전할 것이다. Reflexion [10]이 그 초기 시도이며, 가중치 업데이트 없는 자기 개선이 더욱 정교해질 전망이다.

#### 5.2.2 안전성과 정렬 (Safety and Alignment)

에이전트의 자율성이 증가할수록 안전성 문제가 핵심 과제로 부상한다. 도구 호출의 부작용 제어, 에이전트 행동의 예측 가능성, 인간 가치와의 정렬(alignment), 그리고 샌드박스 탈출(sandbox escape) 방지 등이 중요한 연구 주제이다.

#### 5.2.3 효율적 에이전트 아키텍처

현재 에이전트 시스템은 다수의 LLM 호출로 인한 높은 지연시간(latency)과 비용이 문제이다. 소형 특화 모델의 에이전트 활용, 캐싱 전략, 적응적 추론 깊이 조절(adaptive reasoning depth) 등을 통한 효율성 개선이 필요하다.

#### 5.2.4 에이전트 평가 체계의 성숙

에이전트 성능의 체계적 평가를 위한 벤치마크와 메트릭이 여전히 부족하다. 과제 완료율, 도구 호출 효율성, 추론 정확성, 안전성 등 다차원적 평가 체계의 확립이 요구된다.

#### 5.2.5 분산 Multi-Agent 시스템

대규모 복합 과제를 효율적으로 처리하기 위한 분산 Multi-Agent 시스템의 연구가 활발해질 것이다. 에이전트 간 통신 프로토콜, 작업 할당 최적화, 합의 메커니즘 등이 핵심 과제이다.

---

## 6. 결론

본 레포트에서는 LLM 기반 Agent 아키텍처의 핵심 설계 패턴을 체계적으로 분석하였다. 주요 발견사항을 요약하면 다음과 같다:

**첫째**, ReAct 패턴은 추론과 행동의 교차 실행을 통해 해석 가능하고 유연한 에이전트를 구현하는 기반 패러다임을 확립하였다. Thought-Action-Observation 루프는 이후 대부분의 에이전트 프레임워크가 채택하는 핵심 설계 원리가 되었다.

**둘째**, Tool Use 메커니즘은 Toolformer의 자기 지도 학습 접근에서 시작하여 OpenAI Function Calling의 산업 표준화, MCP의 개방형 프로토콜화, Gorilla의 API 호출 전문화까지 빠르게 진화하였다. 특히 MCP의 등장은 도구 통합의 파편화 문제를 해결하는 중요한 이정표이다.

**셋째**, Plan-and-Execute 패턴은 복잡한 과제의 분해와 체계적 실행에 유리하며, HuggingGPT와 같이 LLM을 컨트롤러로 활용하는 아키텍처의 기반이 되었다.

**넷째**, Multi-Agent 시스템은 AutoGen, CrewAI 등의 프레임워크를 통해 실용적 수준에 도달하였으며, 역할 분담, 대화 기반 협력, 계층적 위임 등 다양한 설계 패턴이 검증되었다.

**다섯째**, 메모리 아키텍처는 단기-장기 기억의 분리, 반성적 기억, 계층적 기억 등으로 정교화되고 있으며, RAG와의 통합을 통해 에이전트의 지식 확장성이 크게 향상되었다.

향후 LLM Agent 연구는 자율적 자기 개선, 안전성 보장, 효율성 최적화, 그리고 분산 다중 에이전트 시스템의 방향으로 진전될 것이다. 특히 에이전틱 코딩의 급속한 발전은 소프트웨어 개발 패러다임 자체의 변화를 예고하고 있으며, 이는 연구자와 실무자 모두에게 새로운 기회와 도전을 제시한다.

---

## 7. 참고문헌

[1] L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen, J. Tang, X. Chen, Y. Lin, W. X. Zhao, Z. Wei, and J.-R. Wen, "A Survey on Large Language Model based Autonomous Agents," *Frontiers of Computer Science*, 2024. arXiv:2308.11432. [Online]. Available: https://arxiv.org/abs/2308.11432

[2] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. Le, and D. Zhou, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," in *Proc. NeurIPS*, 2022. arXiv:2201.11903. [Online]. Available: https://arxiv.org/abs/2201.11903

[3] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, "ReAct: Synergizing Reasoning and Acting in Language Models," in *Proc. ICLR*, 2023. arXiv:2210.03629. [Online]. Available: https://arxiv.org/abs/2210.03629

[4] T. Schick, J. Dwivedi-Yu, R. Dessi, R. Raileanu, M. Lomeli, L. Zettlemoyer, N. Cancedda, and T. Scialom, "Toolformer: Language Models Can Teach Themselves to Use Tools," in *Proc. NeurIPS*, 2023. arXiv:2302.04761. [Online]. Available: https://arxiv.org/abs/2302.04761

[5] S. G. Patil, T. Zhang, X. Wang, and J. E. Gonzalez, "Gorilla: Large Language Model Connected with Massive APIs," arXiv:2305.15334, 2023. [Online]. Available: https://arxiv.org/abs/2305.15334

[6] L. Wang, W. Xu, Y. Lan, Z. Hu, Y. Lan, R. K.-W. Lee, and E.-P. Lim, "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models," in *Proc. ACL*, 2023. arXiv:2305.04091. [Online]. Available: https://arxiv.org/abs/2305.04091

[7] Y. Shen, K. Song, X. Tan, D. Li, W. Lu, and Y. Zhuang, "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face," in *Proc. NeurIPS*, 2023. arXiv:2303.17580. [Online]. Available: https://arxiv.org/abs/2303.17580

[8] Q. Wu, G. Bansal, J. Zhang, Y. Wu, B. Li, E. Zhu, L. Jiang, X. Zhang, S. Zhang, J. Liu, A. H. Awadallah, R. W. White, D. Burger, and C. Wang, "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation," arXiv:2308.08155, 2023. [Online]. Available: https://arxiv.org/abs/2308.08155

[9] J. S. Park, J. C. O'Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein, "Generative Agents: Interactive Simulacra of Human Behavior," in *Proc. ACM UIST*, 2023. arXiv:2304.03442. [Online]. Available: https://arxiv.org/abs/2304.03442

[10] N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, and S. Yao, "Reflexion: Language Agents with Verbal Reinforcement Learning," in *Proc. NeurIPS*, 2023. arXiv:2303.11366. [Online]. Available: https://arxiv.org/abs/2303.11366

[11] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W.-t. Yih, T. Rocktaschel, S. Riedel, and D. Kiela, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," in *Proc. NeurIPS*, 2020. arXiv:2005.11401. [Online]. Available: https://arxiv.org/abs/2005.11401
