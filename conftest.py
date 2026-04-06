"""
Root conftest.py — ensures the project root is on sys.path so that
`pytest tests/ -v` (without `python -m pytest`) can import scheduling_env.
"""
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))
