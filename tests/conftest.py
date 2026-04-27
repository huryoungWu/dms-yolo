import sys
import os

# Add project root to sys.path so tests can import from all modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis import settings

# CI profile: more examples for thorough testing
settings.register_profile("ci", max_examples=200)
# Dev profile: fewer examples for faster iteration
settings.register_profile("dev", max_examples=100)
# Default to dev profile
settings.load_profile("dev")
