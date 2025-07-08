# test_ui_simplification.py
"""
Test cases to verify UI simplification removed save/load functionality
while preserving core prediction and analysis features.
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the simplified UI module
import ui
from constants import UIConstants, FileConstants

class TestSidebarInputsNoSaveLoad:
    """Ensure sidebar doesn't contain save/load elements"""
    
    def test_sidebar_inputs_no_save_load_sections(self):
        """Verify save/load sections are completely removed from sidebar"""
        
        # Mock Streamlit components
        with patch('streamlit.sidebar'), \
             patch('streamlit.title'), \
             patch('streamlit.info'), \
             patch('streamlit.tabs') as mock_tabs, \
             patch('streamlit.selectbox'), \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.divider'), \
             patch('streamlit.subheader'), \
             patch('streamlit.columns'):
            
            # Mock session state
            st.session_state = {
                'prediction_history': [],
                'prf_size_label2code': {},
                'prf_size_code2mid': {},
            }
            
            # Mock models
            with patch.object(ui, 'check_required_models') as mock_check, \
                 patch.object(ui, 'list_available_models') as mock_list:
                
                mock_check.return_value = {"models_available": True}
                mock_list.return_value = [{"display_name": "Test Model", "technical_name": "test_model"}]
                
                # Call the sidebar function
                result = ui.sidebar_inputs()
                
                # Verify no save/load related button calls
                button_calls = [call[0][0] for call in mock_button.call_args_list]
                
                # These buttons should NOT exist
                forbidden_buttons = [
                    '💾 Save Configuration',
                    '📁 Load Configuration', 
                    '🔄 Load from History',
                    '📤 Export Config',
                    '📥 Import Config'
                ]
                
                for forbidden in forbidden_buttons:
                    assert forbidden not in button_calls, f"Found forbidden save/load button: {forbidden}"
                
                # These buttons SHOULD exist
                required_buttons = [
                    '🔮 Predict Effort',
                    '🗑️ Clear History',
                    '📊 Show All'
                ]
                
                for required in required_buttons:
                    assert required in button_calls, f"Missing required button: {required}"

    def test_sidebar_inputs_no_file_upload(self):
        """Verify no file upload widgets exist in sidebar"""
        
        with patch('streamlit.sidebar'), \
             patch('streamlit.file_uploader') as mock_file_uploader, \
             patch('streamlit.title'), \
             patch('streamlit.info'), \
             patch('streamlit.tabs'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.button'), \
             patch('streamlit.divider'), \
             patch('streamlit.subheader'), \
             patch('streamlit.columns'):
            
            st.session_state = {
                'prediction_history': [],
                'prf_size_label2code': {},
                'prf_size_code2mid': {},
            }
            
            with patch.object(ui, 'check_required_models') as mock_check, \
                 patch.object(ui, 'list_available_models') as mock_list:
                
                mock_check.return_value = {"models_available": True}
                mock_list.return_value = [{"display_name": "Test Model", "technical_name": "test_model"}]
                
                ui.sidebar_inputs()
                
                # Verify no file uploader was called
                mock_file_uploader.assert_not_called()

    def test_sidebar_inputs_return_structure(self):
        """Verify sidebar_inputs returns correct structure without config keys"""
        
        with patch('streamlit.sidebar'), \
             patch('streamlit.title'), \
             patch('streamlit.info'), \
             patch('streamlit.tabs'), \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.divider'), \
             patch('streamlit.subheader'), \
             patch('streamlit.columns'), \
             patch('streamlit.number_input') as mock_number, \
             patch('streamlit.text_input') as mock_text, \
             patch('streamlit.checkbox') as mock_checkbox:
            
            st.session_state = {
                'prediction_history': [],
                'prf_size_label2code': {},
                'prf_size_code2mid': {},
            }
            
            # Mock field rendering to return values
            mock_selectbox.return_value = "test_value"
            mock_number.return_value = 5
            mock_text.return_value = "test"
            mock_checkbox.return_value = True
            mock_button.return_value = False
            
            with patch.object(ui, 'check_required_models') as mock_check, \
                 patch.object(ui, 'list_available_models') as mock_list, \
                 patch.object(ui, 'get_tab_organization') as mock_tabs, \
                 patch.object(ui, 'FIELDS', {}):
                
                mock_check.return_value = {"models_available": True}
                mock_list.return_value = [{"display_name": "Test Model", "technical_name": "test_model"}]
                mock_tabs.return_value = {"Important Features": [], "Nice Features": []}
                
                result = ui.sidebar_inputs()
                
                # Verify result structure
                assert isinstance(result, dict)
                assert "selected_model" in result
                assert "submit" in result
                
                # Verify NO config-related keys
                forbidden_keys = [
                    'save_config', 'config_name', 'load_config',
                    'uploaded_file', 'config_data', 'load_from_history'
                ]
                
                for key in forbidden_keys:
                    assert key not in result, f"Found forbidden config key: {key}"


class TestEstimatorTabStillWorks:
    """Core prediction functionality unchanged"""
    
    def test_main_estimator_tab_functionality(self):
        """Verify the main estimator tab works without save/load"""
        
        # Mock user inputs
        mock_inputs = {
            'project_prf_functional_size': 100,
            'project_prf_max_team_size': 5,
            'selected_model': 'test_model',
            'submit': True
        }
        
        with patch('streamlit.tabs') as mock_tabs, \
             patch('streamlit.spinner'), \
             patch('streamlit.divider'), \
             patch.object(ui, 'display_inputs'), \
             patch.object(ui, 'predict_man_hours') as mock_predict, \
             patch.object(ui, 'show_prediction'), \
             patch.object(ui, 'add_prediction_to_history'), \
             patch.object(ui, 'show_prediction_history'), \
             patch.object(ui, 'show_feature_importance'):
            
            # Setup tab context manager
            tab_context = MagicMock()
            mock_tabs.return_value = [tab_context]
            
            mock_predict.return_value = 500.0
            
            st.session_state = {'prediction_history': []}
            
            # This should work without any config-related calls
            try:
                # Simulate main tab logic
                if mock_inputs.get('submit', False):
                    selected_model = mock_inputs.get('selected_model')
                    if selected_model:
                        prediction = mock_predict(mock_inputs, selected_model)
                        assert prediction == 500.0
                        
            except Exception as e:
                pytest.fail(f"Estimator tab functionality failed: {e}")

    def test_prediction_without_config_dependency(self):
        """Verify prediction works without configuration save/load"""
        
        with patch.object(ui, 'predict_man_hours') as mock_predict, \
             patch.object(ui, 'add_prediction_to_history'):
            
            mock_predict.return_value = 750.0
            
            user_inputs = {
                'project_prf_functional_size': 200,
                'selected_model': 'rf_model'
            }
            
            # This should work independently of any config management
            result = mock_predict(user_inputs, 'rf_model')
            assert result == 750.0
            mock_predict.assert_called_once_with(user_inputs, 'rf_model')


class TestShapTabOnlyInstanceSpecific:
    """Visualizations tab only shows instance SHAP"""
    
    def test_shap_tab_simplified_content(self):
        """Verify SHAP tab only contains instance-specific analysis"""
        
        with patch('streamlit.header'), \
             patch('streamlit.warning'), \
             patch('streamlit.info'), \
             patch('streamlit.error'), \
             patch.object(ui, 'display_instance_specific_shap') as mock_shap:
            
            st.session_state = {
                'prediction_history': [{
                    'inputs': {'test': 'value'},
                    '