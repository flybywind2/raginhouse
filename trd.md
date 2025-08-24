# RAG Agent TRD (Technical Requirements Document)

## 1. 시스템 개요
- 목적: Appendix 예시(`internal_llm.py`, `rag_input.py`, `rag_retrieve.py`)와 호환되는 RAG 에이전트 구현.
- 구조: Client → RAG Service(Agent) → Retriever API(ES/Vector) → LLM Gateway → 응답.

## 2. 아키텍처
- 컴포넌트
  - Query Preprocessor: 언어 감지, 정규화, optional 쿼리 확장.
  - Retriever Client: `POST /retrieve-rrf|bm25|knn|cc` 호출, 필터/권한 적용.
  - Evidence Ranker: RRF/스코어 기준 재정렬, 중복 제거.
  - Context Assembler: 토큰 한도 내 컨텍스트 구성, 하이라이트/메타 포함.
  - Prompt Builder: 역할/지시사항/컨텍스트/질문/출처 규격화.
  - LLM Client: `internal_llm.py` 기반, base_url/headers/model 동적 구성.
  - Postprocessor: 출처 주석, 포맷 변환(JSON/Markdown), 안전 필터.
  - AuthN/Z: `x-dep-ticket`, `api-key`, `permission_groups` 강제.
  - Observability: 로깅/트레이싱/메트릭(지연, 결과수, 토큰량).

## 2-0. FastAPI Web UI(최소)
- 구성
  - FastAPI 애플리케이션 + Jinja2 템플릿(간단한 폼/결과 페이지) 또는 단일 HTML + Fetch API.
  - 엔드포인트: `GET /`(질의 입력 폼), `POST /ask`(질의 처리), `POST /feedback`(피드백 제출). 옵션: `GET /ask/stream`(SSE/Chunked 전송).
  - 정적 리소스: 최소 수준(CSS/JS 한 파일), 번들링 없이 제공.
- 동작
  - `/`에서 질의, 인덱스, 리트리버, 결과 수를 입력 → `/ask` 호출 → 응답/근거/점수 렌더링.
  - 각 응답 카드에 좋아요/싫어요 + 사유 입력 → `/feedback` 전송.
- 비고
  - MVP에서는 인증/권한 체크 미적용. 내부 네트워크/개발 환경에서만 노출.

## 2-1. LangGraph 설계
- 상태 스키마(State)
  - `trace_id`, `user_id`, `index_name`, `retriever_cfg`, `query_raw`, `queries_expanded[]`
  - `retrieval_results[]`(bm25, knn, cc), `reranked[]`, `context`, `answer_draft`, `answer_final`
  - `citations[]`, `latency_map{node:ms}`, `feedback_hint`(optional)
- 노드 구성
  - `QueryRewrite`: 다중 쿼리 확장(MQE), stopword 제거, 언어 정규화
  - `RetrieveBM25` | `RetrieveKNN` | `RetrieveCC`: 병렬 실행, 시간 초과 시 부분 결과 사용
  - `FuseAndRerank`: RRF 융합 → MMR 중복 제거 → (옵션) 크로스 인코더 재랭킹
  - `AssembleContext`: 토큰 예산 내 유틸리티 극대화, 하이라이트 포함
  - `Generate`: LLM 생성, 근거 인용 강제
  - `Critique`: 자기평가(근거 일치/사실성/정책 위반) → 수정 필요 플래그
  - `Refine`: 부족 근거 재조회 또는 답변 재작성
  - `End`: 응답 확정 및 로깅
- 체크포인트/실행
  - LangGraph 체크포인터로 각 노드 입력/출력을 보존(재시도/재개)
  - 이벤트 스트리밍으로 per-node 지연과 중간 산출물 관측

## 2-2. 문서 입력/파싱 파이프라인
- 소스
  - Confluence Data Center: 페이지/블로그/첨부 파일(단발 수집; 대규모 크롤링/증분 동기화는 후속)
  - 파일 업로드: PDF/DOCX/PPTX/XLSX 등 일반 문서
- 파싱
  - `docling`으로 텍스트/섹션/헤더/표/캡션/페이지 메타 추출, 가능한 경우 원본 위치 정보 보존
  - 표는 구조 유지 형태(CSV 유사 또는 셀 병합 메타 포함)로 직렬화
  - 이미지/도형은 캡션/대체텍스트 중심 메타만 유지
- 메타데이터 보강(LLM)
  - 입력: 제목/본문 일부/헤더/표 요약/파일 확장자 등
  - 출력: `additional_field` 문자열(요약/토픽/키워드/언어/PII 플래그/문서 타입 등; flat 직렬화)
- 청킹
  - 구조 인지 청킹: `docling` 헤더/섹션 경계 + 토큰 기반 상한, MMR로 중복 제거
- 업서트
  - `POST /insert-doc` 호출로 인덱스에 병합 저장(섹션 단위 분할 시 다건 업서트)
  - insert payload는 반드시 flat 구조 유지: LLM 생성 메타는 `additional_field` 단일 문자열로 직렬화(예: "summary=... | topics=a,b | lang=ko | pii=false | type=pdf")

## 3. 외부 API 연동(참고)
- 공통 헤더
  - `Content-Type: application/json`
  - `x-dep-ticket: <credential_key>`
  - `api-key: <rag_api_key>` (retriever)

- 문서 적재 `POST /insert-doc` (Appendix `rag_input.py`)
```json
{
  "index_name": "your_index_name",
  "data": {
    "doc_id": "ABCD0001",
    "title": "Sample Document",
    "content": "This is a sample document.",
    "permission_groups": ["user"],
    "created_time": "2025-05-29T17:02:54.917+09:00",
    "additional_field": "some value",
    "summary": "Short synopsis",
    " topics": ["product","release"],
    "lang": "ko",
    "pii": false,
    "type": "pdf"
  },
  "chunk_factor": {
    "logic": "fixed_size",
    "chunk_size": 1024,
    "chunk_overlap": 128,
    "separator": " "
  }
}
```
Note: insert payload는 flat 구조만 허용(중첩 JSON 금지). `additional_field`는 문자열로 직렬화한다.

- 검색 `POST /retrieve-rrf|bm25|knn|cc` (Appendix `rag_retrieve.py`)
```json
{
  "index_name": "your_index_name",
  "permission_groups": ["user"],
  "query_text": "Sample query",
  "num_result_doc": 5,
  "fields_exclude": ["v_merge_title_content"],
  "filter": {"example_field_name": ["png"]}
}
```
Note: retrieve 응답은 Elasticsearch Search API 스타일이며, `hits.hits` 배열에 문서가 포함된다.
예시 응답(축약):
```json
{
  "took": 12,
  "timed_out": false,
  "hits": {
    "total": { "value": 123, "relation": "eq" },
    "max_score": 3.21,
    "hits": [
      {
        "_index": "your_index_name",
        "_id": "ABCD0001#000",
        "_score": 3.21,
        "_source": {
          "doc_id": "ABCD0001",
          "title": "Sample Document",
          "content": "...chunk text...",
          "permission_groups": ["user"],
          "created_time": "2025-05-29T17:02:54.917+09:00",
          "additional_field": "summary=...; topics=..."
        }
      }
    ]
  }
}
```
에이전트는 `_score`를 점수로 사용하고, `citations`는 각 hit의 `_source.title`, `_source.doc_id`, 스니펫(컨텍스트)에서 구성한다.

## 4. Agent API 설계(내부)
- `POST /ask`
  - 요청
    - `query_text` (string, required)
    - `index_name` (string, required)
    - `permission_groups` (string[], required)
    - `retriever` (enum: rrf|bm25|knn|cc, default=rrf)
    - `num_result_doc` (int, default=5)
    - `model` (string, optional) — 예: `llama4 maverick`
    - `answer_format` (enum: markdown|json, default=markdown)
  - 응답
    - `answer` (string|object)
    - `citations` ([{`doc_id`,`title`,`score`,`snippet`}])
    - `latency_ms` (number)
    - `trace_id` (string)
    - `debug` (object?) — 실험/튜닝 시 LangGraph 노드 지연/선택 요약(옵션)

## 4-1. Feedback API 설계(내부)
- `POST /feedback`
  - 요청: `trace_id`(string, required), `rating`(enum: up|down), `reason`(string?), `proposed_answer`(string?), `selected_citations`(string[]?), `tags`(string[]?)
  - 응답: `{ status: "ok" }`
- `GET /feedback/metrics`
  - 응답: `{ positive_rate, counts_by_reason, ndcg@k, mrr@k, trending_queries[] }`
- 저장 모델
  - 테이블: `feedback(trace_id PK, user_id, rating, reason, proposed_answer, citations[], created_at)`
  - 개인정보/민감정보 마스킹: MVP 단계에서는 미적용(운영 전 단계에서 추가 예정)

## 4-2. Ingest API(내부, MVP)
- `POST /ingest/file`
  - form-data: `file`(required), `index_name`(string, required), `permission_groups[]`(optional)
  - 동작: `docling` 파싱 → LLM 메타 보강(`additional_field` 문자열 생성) → `insert-doc` 호출(섹션 분할 시 다건 업서트)
- `POST /ingest/confluence`
  - body: `{ base_url, page_id, index_name }` (pageId 기반만 지원)
  - 동작: Confluence DC REST로 pageId 단발 수집 → 동일 파이프라인 적용
  - 네트워크: SSL 검증 비활성화(verify=False)로 호출(MVP 한정)

## 5. LLM 연동
- ChatOpenAI 래퍼(`internal_llm.py`) 사용.
- 환경/헤더
  - `OPENAI_API_KEY`, `LLM_BASE_URL`, `MODEL_NAME`
  - 기본 헤더: `x-dep-ticket`, `Send-System-Name`, `User-ID`, `User-Type`, `Prompt-Msg-Id`, `Completion-Msg-Id`(UUID).
- 타임아웃: 15s 기본, 재시도(지수 백오프, 2회).

## 5-1. 재랭킹/쿼리 재작성/PRF 세부
- 쿼리 재작성(MQE): LLM으로 의미 등가 쿼리 n개 생성 → RRF 융합
- PRF: 상위 k 문서 키워드 추출 → 재질의에 반영
- 재랭킹: 크로스 인코더 API(내부/외부)로 상위 N 후보를 정밀 재정렬(시간 상한 설정)

## 6. 구성(환경 변수)
- `RAG_BASE_URL` (예: `http://localhost:8000`)
- `RAG_API_KEY`
- `DEP_TICKET`
- `INDEX_NAME` (기본값)
- `RETRIEVER` (기본 rrf)
- `OPENAI_API_KEY`
- `LLM_BASE_URL`
- `MODEL_NAME`
 - `DOCLING_ENABLED=true`
 - `CONFLUENCE_BASE_URL`, `CONFLUENCE_TOKEN`(MVP에서는 로컬 환경 변수만 사용)

## 6-1. 캐시 설계
- 의미 캐시: `hash(normalize(query_text), permission_groups, index, retriever_cfg)` → 컨텍스트/응답 TTL 30~120s
- 피드백 힌트 캐시: 동일/유사 쿼리에 최근 피드백 태그를 힌트로 제공
- 무효화: 문서 업데이트/재인덱싱 시 관련 키 범위 무효화

## 7. 프롬프트 설계(요약)
- 지시: “제공된 근거만을 사용해 답변하고, 확실치 않으면 모른다고 답하라. 각 근거의 `doc_id`를 각주로 표기하라.”
- 입력: 사용자 질문, 정제된 근거 n개(제목/스니펫/중요도), 제한 토큰 내 구성.
- 출력: markdown 기본, 필요한 경우 JSON 스키마 강제.

## 8. 오류 처리/리트라이
- Retriever HTTP 오류: 2xx 외 상태 → 재시도(최대 2), 실패 시 빈 근거로 안전 응답.
- LLM 타임아웃/에러: 짧은 요약 모드로 Fallback.
- 빈 검색 결과: 사용자 재질의 가이드 제공.

## 9. 성능/최적화
- 동시 HTTP(검색 샤드/유형 병렬 호출), RRF 융합.
- 스니펫 길이/개수 제한으로 토큰 예산 관리.
- 결과 캐시(쿼리 키 기반, 30~120s TTL) 옵션.

## 9-1. 실험/적응(Online)
- A/B 실험: 프롬프트/리트리버/재랭커 조합을 코호트로 분리
- 멀티암 밴딧(선택): 클릭/피드백 양의율을 보상으로 가중 업데이트
- 피드백 반영: 특정 문서/필드 가중치 보정, 쿼리 템플릿 미세조정

## 10. 로깅/관측
- 구조화 로그: `trace_id`, `user_id`, `index`, `retriever`, 결과 수, 지연.
- 메트릭: 검색/생성 지연, 토큰 사용량, 오류율.
- 트레이싱: Retriever/LLM 호출 span; `Prompt-Msg-Id`/`Completion-Msg-Id` 연동.
 - LangGraph 노드별 지연/성공률, 재시도/분기 비중, 캐시 적중률 수집.

## 11. 보안
- MVP: 보안/인증 미적용(내부 개발/테스트 환경 한정 서비스).
- 후속: SSO/JWT, RBAC, 요청 서명/레이트리밋, PII 마스킹/암호화.

## 12. 데이터 모델(요약)
- 문서 메타: `doc_id`, `title`, `created_time`, `permission_groups`, 기타 필드.
- 컨텐츠: `content`(원문) → 청킹(`chunk_size`, `overlap`, `separator`).
- 검색 결과: `{doc_id, score, snippet, fields...}`

## 13. 테스트 전략
- 유닛: Prompt Builder, Context Assembler, 스코어 융합.
- 통합: Retriever 모킹/로컬 엔드포인트, 권한 필터 케이스.
- 회귀: 샘플 질의 스냅샷 비교, 금지어/안전 필터.
- 부하: 동시 50~100 RPS에서 P95 확인.

## 14. 배포/운영
- 컨테이너(이미지 빌드) → Staging → Prod.
- 구성 주입: 환경 변수/시크릿 매니저.
- 롤백: 블루/그린 또는 카나리.

## 15. 오픈 이슈
- 재랭킹 모델(크로스 인코더) 투입 시 비용/지연.
- 인덱스 스키마 표준화 범위.
- 다국어 쿼리 확장/번역 전략.
