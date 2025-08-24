#!/usr/bin/env python3
"""
Development server runner for RAG Agent

초보자용 설명:
- 개발 환경에서 FastAPI 서버를 실행하는 진입점입니다.
- `python run_dev.py`로 실행하면 자동 리로드(reload)가 켜진 상태로 서버가 뜹니다.
"""
import os
import sys
import logging
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

def main():
    # Set environment variables for development
    # 초보자용: PYTHONPATH를 프로젝트 루트로 설정해 모듈 임포트가 편해집니다.
    os.environ.setdefault("PYTHONPATH", str(project_root))
    
    # Configure logging
    # 초보자용: 개발 콘솔에서 보기 쉬운 포맷으로 로깅을 설정합니다.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        import uvicorn
        
        print("Starting RAG Agent Development Server...")
        print("API Documentation: http://localhost:8080/docs")
        print("Web UI: http://localhost:8080/")
        print("Press Ctrl+C to stop")
        print()
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8080,
            reload=True,
            log_level="info"
        )
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
