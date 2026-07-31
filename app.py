import sys
import os

# Add the 'web' directory to the path so that modules like 'service' can be resolved
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "web"))

# Import the Flask app object from web/app.py
from web.app import app

if __name__ == "__main__":
    app.run()
