"""
Mock data and fixtures for UI tests.
"""

import pandas as pd
import numpy as np

def get_sample_user_inputs():
    """Get sample user inputs for testing."""
    return {
        'project_prf_functional_size': 100,
        'project_prf_max_team_size': 5,
        'external_eef_industry_sector': 'Financial',
        'tech_tf_primary_programming_language': 'Java',
        'selected_model': 'rf_model'
    }

def get_sample_prediction_history():
    """Get sample prediction history."""
    return [
        {
            'timestamp': '2024-01-01 10:00:00',
            'model': 'Random Forest',
            'model_technical': 'rf_model', 
            'prediction_hours': 480.0,
            'inputs': get_sample_user_inputs()
        }
    ]
