# RAG Agent LangGraph Workflow Visualization

LangGraph 워크플로우가 성공적으로 시각화되었습니다. 여러 형태의 다이어그램이 생성되었습니다.

## 📊 생성된 시각화 파일들

### 1. **메인 워크플로우 다이어그램**
- **파일**: `rag_workflow_visualization.png`
- **설명**: 전체 RAG 워크플로우의 노드와 엣지 구조를 보여주는 메인 다이어그램
- **특징**: 
  - 11개 노드와 11개 엣지
  - 색상별 노드 분류 (시작/끝, 쿼리 처리, 검색, 융합/랭킹, 컨텍스트 조합, 답변 생성, 답변 평가, 답변 개선)
  - 조건부 라우팅 표시 (점선으로 표시)

### 2. **상세 데이터 플로우 다이어그램**
- **파일**: `rag_workflow_detailed.png`
- **설명**: 데이터의 흐름과 변환 과정을 자세히 보여주는 다이어그램
- **특징**:
  - 각 단계에서 처리되는 데이터 유형 표시
  - 검색 결과의 융합 및 컨텍스트 선택 과정 시각화

### 3. **Mermaid 다이어그램**
- **파일**: `rag_workflow.mmd`
- **설명**: LangGraph 네이티브 Mermaid 형식의 플로우차트
- **용도**: 웹 환경이나 Mermaid 지원 도구에서 렌더링 가능

## 🔄 워크플로우 구조 분석

### 노드 구성 (11개)
```
1. __start__        - 워크플로우 시작점
2. query_rewrite    - 쿼리 재작성 및 확장
3. retrieve_bm25    - BM25 검색
4. retrieve_knn     - kNN 벡터 검색  
5. retrieve_cc      - Column Conditions 검색
6. fuse_and_rerank  - RRF 융합 및 재랭킹
7. assemble_context - MMR 컨텍스트 조합
8. generate_answer  - LLM 답변 생성
9. critique_answer  - 답변 평가
10. refine_answer   - 답변 개선
11. __end__         - 워크플로우 종료점
```

### 엣지 연결 (11개)
```
순차 실행 경로:
START → query_rewrite → retrieve_bm25 → retrieve_knn → retrieve_cc → fuse_and_rerank → assemble_context → generate_answer → critique_answer

조건부 분기:
critique_answer → END (개선 불필요)
critique_answer → refine_answer → END (개선 필요)
```

## 🎯 워크플로우 특징

### ✅ **수정된 구조의 장점**
1. **순차 실행**: BM25 → kNN → CC 순서로 안정적인 검색 실행
2. **상태 누적**: 각 검색 결과가 `state['retrieval_results']`에 순차 저장
3. **조건부 개선**: 답변 품질에 따른 선택적 개선 프로세스
4. **예측 가능한 흐름**: LangGraph 병렬 실행 문제 해결로 일관된 동작 보장

### 🔧 **핵심 프로세스**
1. **멀티 쿼리 확장**: 원본 쿼리를 3개 변형으로 확장
2. **다중 검색**: BM25(키워드), kNN(벡터), CC(컬럼 조건) 순차 실행
3. **RRF 융합**: Reciprocal Rank Fusion으로 검색 결과 통합
4. **MMR 선택**: Maximal Marginal Relevance로 최적 컨텍스트 선택
5. **답변 생성**: 선택된 모델로 답변 생성
6. **품질 평가**: 답변 품질 평가 후 개선 여부 결정

## 🚀 사용 방법

### Mermaid 다이어그램 렌더링
```bash
# Mermaid CLI 설치 (선택사항)
npm install -g @mermaid-js/mermaid-cli

# PNG로 변환
mmdc -i rag_workflow.mmd -o rag_workflow_mermaid.png
```

### 온라인 Mermaid 에디터
- [Mermaid Live Editor](https://mermaid.live/)에서 `rag_workflow.mmd` 내용을 붙여넣어 실시간 렌더링 가능

## 📈 성능 최적화 포인트

1. **검색 최적화**: 각 retriever별 병렬 실행 → 순차 실행으로 변경
2. **메모리 효율성**: 상태 객체를 통한 효율적인 데이터 전달
3. **조건부 실행**: 불필요한 답변 개선 과정 생략 가능
4. **캐싱**: 동일 쿼리에 대한 중간 결과 캐싱 지원

---

**시각화 완료**: 모든 노드와 엣지 연결이 검증되었으며, 워크플로우가 정상적으로 작동합니다. 🎉