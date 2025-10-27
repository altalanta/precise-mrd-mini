#!/usr/bin/env python3
"""Script to run the MRD Pipeline Dashboard with API integration."""

import subprocess
import sys
import time
import threading
import signal
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def run_api_server():
    """Run the API server in a separate process."""
    try:
        from precise_mrd.api import run_api_server
        print("🚀 Starting API server...")
        run_api_server(host="localhost", port=8000)
    except Exception as e:
        print(f"❌ API server failed: {e}")

def run_dashboard():
    """Run the Streamlit dashboard."""
    try:
        from precise_mrd.dashboard import run_dashboard
        print("📊 Starting dashboard...")
        run_dashboard()
    except Exception as e:
        print(f"❌ Dashboard failed: {e}")

def main():
    """Main function to run both API and dashboard."""
    print("🔬 Starting Precise MRD Pipeline Dashboard & API")
    print("=" * 60)

    # Start API server in background thread
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()

    # Give API server time to start
    time.sleep(3)

    # Check if API is running
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
        else:
            print("⚠️  API server may have issues")
    except:
        print("⚠️  Could not verify API server status")

    print("🌐 Dashboard will be available at: http://localhost:8501")
    print("🔌 API endpoints available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("-" * 60)

    # Run dashboard
    try:
        run_dashboard()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        # Cleanup processes
        os._exit(0)

if __name__ == "__main__":
    main()




