#!/usr/bin/env python3
"""
LangGraph Native Visualization
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
    os.environ["OPENAI_API_KEY"] = "dummy-key-for-testing"
    os.environ["RAG_API_KEY"] = "dummy-rag-key"  
    os.environ["DEP_TICKET"] = "dummy-ticket"
    print("Using dummy environment values for testing")

sys.path.insert(0, str(project_root))

def create_native_visualization():
    """Create visualization using LangGraph's built-in methods"""
    try:
        from src.agents.rag_workflow import RAGWorkflow
        
        print("Creating native LangGraph visualization...")
        
        # Initialize workflow
        workflow_instance = RAGWorkflow()
        
        # Get the compiled graph
        graph = workflow_instance.workflow
        
        # Generate Mermaid diagram
        try:
            mermaid_code = graph.get_graph().draw_mermaid()
            
            # Save Mermaid code to file
            mermaid_path = project_root / "rag_workflow.mmd"
            with open(mermaid_path, 'w', encoding='utf-8') as f:
                f.write(mermaid_code)
            
            print(f"Mermaid diagram saved to: {mermaid_path}")
            print("\nMermaid Code:")
            print("-" * 50)
            print(mermaid_code)
            print("-" * 50)
            
        except Exception as e:
            print(f"Mermaid generation failed: {e}")
        
        # Try to generate PNG using draw_ascii
        try:
            ascii_diagram = graph.get_graph().draw_ascii()
            print("\nASCII Diagram:")
            print("=" * 60)
            print(ascii_diagram)
            print("=" * 60)
            
            # Save ASCII diagram
            ascii_path = project_root / "rag_workflow_ascii.txt"
            with open(ascii_path, 'w', encoding='utf-8') as f:
                f.write(ascii_diagram)
            print(f"ASCII diagram saved to: {ascii_path}")
            
        except Exception as e:
            print(f"ASCII diagram generation failed: {e}")
        
        # Print graph information
        print(f"\nGraph Information:")
        print(f"Nodes: {len(graph.get_graph().nodes)}")
        print(f"Edges: {len(graph.get_graph().edges)}")
        
        # Print node details
        print(f"\nNode Details:")
        for node_id, node_data in graph.get_graph().nodes.items():
            print(f"  {node_id}: {type(node_data).__name__}")
        
        # Print edge details
        print(f"\nEdge Details:")
        for edge in graph.get_graph().edges:
            print(f"  {edge.source} -> {edge.target}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_native_visualization()
    if success:
        print("\nNative LangGraph visualization completed!")
    else:
        print("\nNative LangGraph visualization failed!")
        sys.exit(1)