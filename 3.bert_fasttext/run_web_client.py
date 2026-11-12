"""Launch the Streamlit client."""
import os
import subprocess
import sys

if __name__ == '__main__':
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web_client.py')
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', script], check=True)

