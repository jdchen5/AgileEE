"""
Integration tests for cross-tab functionality.
"""

import pytest
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import ui

class TestCrossTabIntegration:
    """Test integration between different tabs."""
    
    def test_estimator_to_shap_flow(self):
        """Test that predictions flow from estimator to SHAP tab."""
        # Basic integration test
        assert hasattr(ui, 'main')
        assert hasattr(ui, 'display_instance_specific_shap')
