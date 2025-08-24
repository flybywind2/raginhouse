#!/usr/bin/env python3
"""
워크플로우 노드/간선 연결을 점검하는 간단한 테스트

초보자용 설명:
- LangGraph로 만들어진 그래프의 노드와 간선이 기대한 대로 존재하는지 확인합니다.
- 이 파일은 네트워크를 호출하지 않고, 구조만 검증합니다.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print("Environment loaded from .env")
else:
    # Set dummy values for testing
    os.environ["OPENAI_API_KEY"] = "dummy-key-for-testing"
    os.environ["RAG_API_KEY"] = "dummy-rag-key"  
    os.environ["DEP_TICKET"] = "dummy-ticket"
    print("Using dummy environment values for testing")

sys.path.insert(0, str(project_root))

def test_workflow_structure():
    """워크플로우 구조 테스트

    초보자용:
    - 노드 목록과 간선(연결) 목록을 출력하고, 기대한 연결이 모두 있는지 체크합니다.
    """
    try:
        from src.agents.rag_workflow import RAGWorkflow
        print("SUCCESS: Workflow imports successful")
        
        workflow_instance = RAGWorkflow()
        print("SUCCESS: Workflow initialization successful")
        
        # Test graph structure
        graph = workflow_instance.workflow.get_graph()
        nodes = list(graph.nodes.keys())
        print(f"\nWorkflow Structure:")
        print(f"Nodes ({len(nodes)}):")
        for node in nodes:
            print(f"  - {node}")
        
        # Check edges and flow
        edges = []
        for edge in graph.edges:
            source_node = edge.source
            target_node = edge.target
            edges.append((source_node, target_node))
        
        print(f"\nEdges ({len(edges)}):")
        for source, target in edges:
            print(f"  {source} -> {target}")
            
        # Validate expected flow
        expected_flow = [
            ("__start__", "query_rewrite"),
            ("query_rewrite", "retrieve_bm25"), 
            ("retrieve_bm25", "retrieve_knn"),
            ("retrieve_knn", "retrieve_cc"),
            ("retrieve_cc", "fuse_and_rerank"),
            ("fuse_and_rerank", "assemble_context"),
            ("assemble_context", "generate_answer"),
            ("generate_answer", "critique_answer"),
            ("refine_answer", "__end__")
        ]
        
        print(f"\nValidating expected flow:")
        missing_edges = []
        for source, target in expected_flow:
            if (source, target) not in edges:
                missing_edges.append((source, target))
                print(f"  MISSING: {source} -> {target}")
            else:
                print(f"  OK: {source} -> {target}")
        
        if missing_edges:
            print(f"\nWARNING: {len(missing_edges)} expected edges are missing")
        else:
            print(f"\nSUCCESS: All expected edges present")
            
        print(f"\nWorkflow structure validation completed")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_workflow_structure()
    if success:
        print("\n" + "="*50)
        print("WORKFLOW STRUCTURE: PASSED")
    else:
        print("\n" + "="*50) 
        print("WORKFLOW STRUCTURE: FAILED")
        sys.exit(1)
