#!/usr/bin/env python3
"""
LangGraph RAG Workflow Visualization
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import networkx as nx

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

def create_workflow_visualization():
    """Create a visual representation of the RAG workflow"""
    
    # Import after setting environment
    from src.agents.rag_workflow import RAGWorkflow
    
    print("Creating RAG workflow visualization...")
    
    # Initialize workflow
    workflow_instance = RAGWorkflow()
    graph = workflow_instance.workflow.get_graph()
    
    # Extract nodes and edges
    nodes = list(graph.nodes.keys())
    edges = [(edge.source, edge.target) for edge in graph.edges]
    
    print(f"Found {len(nodes)} nodes and {len(edges)} edges")
    
    # Create NetworkX graph for layout
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(nodes)
    nx_graph.add_edges_from(edges)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define node positions manually for better layout
    positions = {
        '__start__': (1, 9),
        'query_rewrite': (3, 9),
        'retrieve_bm25': (5, 8.5),
        'retrieve_knn': (5, 7.5),
        'retrieve_cc': (5, 6.5),
        'fuse_and_rerank': (7, 7.5),
        'assemble_context': (7, 5.5),
        'generate_answer': (7, 3.5),
        'critique_answer': (5, 2),
        'refine_answer': (3, 1),
        '__end__': (1, 1)
    }
    
    # Define node colors and styles
    node_colors = {
        '__start__': '#4CAF50',  # Green
        '__end__': '#F44336',    # Red
        'query_rewrite': '#2196F3',  # Blue
        'retrieve_bm25': '#FF9800',  # Orange
        'retrieve_knn': '#FF9800',   # Orange
        'retrieve_cc': '#FF9800',    # Orange
        'fuse_and_rerank': '#9C27B0',  # Purple
        'assemble_context': '#607D8B',  # Blue Grey
        'generate_answer': '#795548',   # Brown
        'critique_answer': '#E91E63',   # Pink
        'refine_answer': '#FFC107'      # Amber
    }
    
    # Define node labels
    node_labels = {
        '__start__': 'START',
        '__end__': 'END', 
        'query_rewrite': 'Query\\nRewrite',
        'retrieve_bm25': 'BM25\\nRetrieval',
        'retrieve_knn': 'kNN\\nRetrieval', 
        'retrieve_cc': 'CC\\nRetrieval',
        'fuse_and_rerank': 'RRF Fusion\\n& Rerank',
        'assemble_context': 'Assemble\\nContext',
        'generate_answer': 'Generate\\nAnswer',
        'critique_answer': 'Critique\\nAnswer',
        'refine_answer': 'Refine\\nAnswer'
    }
    
    # Draw nodes
    node_patches = {}
    for node in nodes:
        x, y = positions[node]
        color = node_colors.get(node, '#CCCCCC')
        
        # Create rounded rectangle
        if node in ['__start__', '__end__']:
            # Special styling for start/end nodes
            patch = FancyBboxPatch(
                (x-0.3, y-0.2), 0.6, 0.4,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor='black',
                linewidth=2
            )
        else:
            # Regular nodes
            patch = FancyBboxPatch(
                (x-0.4, y-0.25), 0.8, 0.5,
                boxstyle="round,pad=0.05", 
                facecolor=color,
                edgecolor='black',
                linewidth=1,
                alpha=0.8
            )
        
        ax.add_patch(patch)
        node_patches[node] = patch
        
        # Add text label
        ax.text(x, y, node_labels.get(node, node), 
               ha='center', va='center', 
               fontsize=9, fontweight='bold', 
               color='white' if node in ['__start__', '__end__'] else 'black')
    
    # Draw edges
    for source, target in edges:
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        
        # Determine edge style
        if source == 'critique_answer':
            # Conditional edges
            if target == '__end__':
                edge_color = 'green'
                edge_style = '--'
                label = 'no refine'
            else:  # target == 'refine_answer'
                edge_color = 'orange' 
                edge_style = '--'
                label = 'refine'
        else:
            edge_color = 'black'
            edge_style = '-'
            label = None
        
        # Create arrow
        arrow = ConnectionPatch(
            (x1, y1), (x2, y2), "data", "data",
            arrowstyle="->", shrinkA=25, shrinkB=25,
            mutation_scale=20, fc=edge_color, ec=edge_color,
            linestyle=edge_style, linewidth=2
        )
        ax.add_patch(arrow)
        
        # Add edge label for conditional edges
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, label, fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8),
                   ha='center', va='center')
    
    # Add title and description
    ax.text(5, 9.7, 'RAG Agent Workflow (LangGraph)', 
           ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#4CAF50', label='Start/End'),
        mpatches.Patch(color='#2196F3', label='Query Processing'),
        mpatches.Patch(color='#FF9800', label='Retrieval'),
        mpatches.Patch(color='#9C27B0', label='Fusion/Ranking'),
        mpatches.Patch(color='#607D8B', label='Context Assembly'),
        mpatches.Patch(color='#795548', label='Answer Generation'),
        mpatches.Patch(color='#E91E63', label='Answer Critique'),
        mpatches.Patch(color='#FFC107', label='Answer Refinement')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Add workflow description
    description = """
    Sequential RAG Workflow:
    1. Query rewriting and expansion
    2. Multi-retriever execution (BM25 → kNN → CC)  
    3. RRF fusion and reranking
    4. Context assembly with MMR selection
    5. LLM answer generation
    6. Answer critique and conditional refinement
    """
    
    ax.text(0.5, 0.3, description, fontsize=10, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7),
           verticalalignment='top')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = project_root / "rag_workflow_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Workflow visualization saved to: {output_path}")
    
    return output_path

def create_detailed_flow_diagram():
    """Create a detailed flow diagram with data flow"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define detailed flow steps
    flow_steps = [
        {"name": "User Query", "pos": (1, 9), "color": "#4CAF50"},
        {"name": "Multi-Query\\nExpansion", "pos": (3, 9), "color": "#2196F3"},
        {"name": "BM25\\nSearch", "pos": (5, 8.5), "color": "#FF9800"},
        {"name": "Vector kNN\\nSearch", "pos": (5, 7), "color": "#FF9800"},
        {"name": "Column\\nConditions", "pos": (5, 5.5), "color": "#FF9800"}, 
        {"name": "RRF Fusion", "pos": (7.5, 7), "color": "#9C27B0"},
        {"name": "MMR Context\\nSelection", "pos": (9.5, 7), "color": "#607D8B"},
        {"name": "LLM Answer\\nGeneration", "pos": (9.5, 4.5), "color": "#795548"},
        {"name": "Answer\\nCritique", "pos": (7.5, 2.5), "color": "#E91E63"},
        {"name": "Answer\\nRefinement", "pos": (5, 1.5), "color": "#FFC107"},
        {"name": "Final\\nResponse", "pos": (2, 1.5), "color": "#4CAF50"}
    ]
    
    # Draw flow steps
    for step in flow_steps:
        x, y = step["pos"]
        color = step["color"]
        
        # Draw node
        patch = FancyBboxPatch(
            (x-0.4, y-0.3), 0.8, 0.6,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=1,
            alpha=0.8
        )
        ax.add_patch(patch)
        
        # Add text
        ax.text(x, y, step["name"], ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')
    
    # Draw flow arrows
    flow_connections = [
        (0, 1), (1, 2), (2, 5), (1, 3), (3, 5), (1, 4), (4, 5),  # Retrieval paths
        (5, 6), (6, 7), (7, 8),  # Main flow
        (8, 9), (9, 10), (8, 10)  # Conditional paths
    ]
    
    for src_idx, tgt_idx in flow_connections:
        src_pos = flow_steps[src_idx]["pos"]
        tgt_pos = flow_steps[tgt_idx]["pos"]
        
        # Special handling for conditional arrows
        if src_idx == 8 and tgt_idx == 10:  # critique -> final (no refinement)
            arrow_color = 'green'
            arrow_style = '--'
        elif src_idx == 8 and tgt_idx == 9:  # critique -> refinement
            arrow_color = 'orange'
            arrow_style = '--'
        else:
            arrow_color = 'black'
            arrow_style = '-'
        
        arrow = ConnectionPatch(
            src_pos, tgt_pos, "data", "data",
            arrowstyle="->", shrinkA=30, shrinkB=30,
            mutation_scale=15, fc=arrow_color, ec=arrow_color,
            linestyle=arrow_style, linewidth=2
        )
        ax.add_patch(arrow)
    
    # Add title
    ax.text(6, 9.5, 'RAG Agent Data Flow Diagram', 
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Add data flow annotations
    ax.text(1, 8.3, "Original Query\n+ Expanded Variants", fontsize=8, 
           ha='center', bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow'))
    
    ax.text(5, 4, "Retrieved\nDocuments", fontsize=8,
           ha='center', bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral'))
    
    ax.text(7.5, 5.5, "Fused &\nRanked Results", fontsize=8,
           ha='center', bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen'))
    
    ax.text(9.5, 5.8, "Selected\nContext", fontsize=8,
           ha='center', bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue'))
    
    plt.tight_layout()
    
    # Save detailed flow diagram
    output_path = project_root / "rag_workflow_detailed.png" 
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Detailed flow diagram saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    try:
        print("Starting LangGraph visualization...")
        
        # Create main workflow visualization
        main_viz = create_workflow_visualization()
        
        # Create detailed flow diagram
        detailed_viz = create_detailed_flow_diagram()
        
        print("\nVisualization completed successfully!")
        print(f"Main workflow: {main_viz}")
        print(f"Detailed flow: {detailed_viz}")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)