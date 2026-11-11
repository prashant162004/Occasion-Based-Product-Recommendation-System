import os
import sys
import uvicorn
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("[INFO] Starting Occasion-Based Product Recommendation System...")
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "127.0.0.1")
    
    print(f"[INFO] Server address: http://{host}:{port}")
    print("[INFO] Press Ctrl+C to stop.")
    
    reload = os.getenv("RELOAD", "false").lower() in ("true", "1")
    
    uvicorn.run(
        "backend.app.main:app",
        host=host,
        port=port,
        reload=reload
    )

if __name__ == "__main__":
    main()
