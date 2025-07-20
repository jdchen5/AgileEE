# run_agileee.py - Simple launcher
import os
import sys

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Now run streamlit
if __name__ == "__main__":
    import subprocess
    app_path = "agileee/main.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])