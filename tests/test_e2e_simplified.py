# Create a new file: tests/test_e2e_simplified.py
import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agileee.ui as ui

class TestE2ESimplified:
    """Simplified end-to-end tests"""
    
    def setup_method(self):
        """Setup for each test"""
        st.session_state = {
            'prediction_history': [],
            'current_prediction_results': None,
            'form_attempted': False,
            'prf_size_label2code': {},
            'prf_size_code2mid': {},
        }

    def test_basic_prediction_workflow(self):
        """Test basic prediction workflow"""
        
        with patch.object(ui, 'predict_man_hours') as mock_predict:
            mock_predict.return_value = 480.0
            
            # Test prediction
            user_inputs = {
                'project_prf_functional_size': 100,
                'project_prf_max_team_size': 5
            }
            
            prediction = ui.predict_man_hours(user_inputs, 'rf_model')
            assert prediction == 480.0
            
            # Test adding to history
            ui.add_prediction_to_history(user_inputs, 'rf_model', prediction)
            assert len(st.session_state['prediction_history']) == 1

    def test_model_comparison_basic(self):
        """Test basic model comparison"""
        
        # Add test data
        st.session_state['prediction_history'] = [
            {'model_technical': 'rf_model', 'prediction_hours': 480.0},
            {'model_technical': 'xgb_model', 'prediction_hours': 520.0}
        ]
        
        with patch('streamlit.header'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.dataframe'), \
             patch('plotly.express.box'), \
             patch('streamlit.plotly_chart'):
            
            mock_columns.return_value = [MagicMock(), MagicMock()]
            
            # Should not crash
            ui.display_model_comparison()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])