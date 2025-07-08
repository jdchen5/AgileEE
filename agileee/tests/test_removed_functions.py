# test_removed_functions.py
"""
Test cases to verify that configuration save/load functions are completely removed
and that the UI simplification was successful.
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os
import inspect

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the simplified UI module
import ui
from constants import UIConstants, FileConstants

class TestConfigFunctionsRemoved:
    """Verify config save/load functions are completely removed"""
    
    def test_make_current_config_json_removed(self):
        """Verify make_current_config_json function is removed"""
        
        # This function should not exist in the simplified UI
        assert not hasattr(ui, 'make_current_config_json'), \
            "make_current_config_json function should be removed"

    def test_make_history_json_removed(self):
        """Verify make_history_json function is removed"""
        
        # This function should not exist in the simplified UI
        assert not hasattr(ui, 'make_history_json'), \
            "make_history_json function should be removed"

    def test_load_configuration_from_data_removed(self):
        """Verify load_configuration_from_data function is removed"""
        
        # This function should not exist in the simplified UI
        assert not hasattr(ui, 'load_configuration_from_data'), \
            "load_configuration_from_data function should be removed"

    def test_save_configuration_removed(self):
        """Verify save_configuration function is removed"""
        
        # This function should not exist in the simplified UI
        assert not hasattr(ui, 'save_configuration'), \
            "save_configuration function should be removed"

    def test_load_configuration_removed(self):
        """Verify load_configuration function is removed"""
        
        # This function should not exist in the simplified UI
        assert not hasattr(ui, 'load_configuration'), \
            "load_configuration function should be removed"

    def test_export_configuration_removed(self):
        """Verify export_configuration function is removed"""
        
        # This function should not exist in the simplified UI
        assert not hasattr(ui, 'export_configuration'), \
            "export_configuration function should be removed"

    def test_import_configuration_removed(self):
        """Verify import_configuration function is removed"""
        
        # This function should not exist in the simplified UI
        assert not hasattr(ui, 'import_configuration'), \
            "import_configuration function should be removed"

class TestFileUploadUIRemoved:
    """Verify file upload UI elements are completely removed"""
    
    def test_no_file_uploader_in_sidebar(self):
        """Verify no file uploader widgets in sidebar_inputs"""
        
        with patch('streamlit.file_uploader') as mock_file_uploader, \
             patch('streamlit.sidebar'), \
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

    def test_no_json_upload_widgets(self):
        """Verify no JSON upload widgets exist"""
        
        # Search through all UI functions to ensure no JSON upload widgets
        ui_functions = [getattr(ui, name) for name in dir(ui) if callable(getattr(ui, name))]
        
        # Check function source code for file upload patterns
        for func in ui_functions:
            if hasattr(func, '__code__'):
                try:
                    source = inspect.getsource(func)
                    forbidden_patterns = [
                        'file_uploader',
                        'upload_json',
                        'json_uploader',
                        'accept_multiple_files'
                    ]
                    
                    for pattern in forbidden_patterns:
                        assert pattern not in source, \
                            f"Found forbidden upload pattern '{pattern}' in function {func.__name__}"
                            
                except (OSError, TypeError):
                    # Skip if source is not available
                    pass

    def test_no_download_buttons(self):
        """Verify no download buttons for config files"""
        
        with patch('streamlit.download_button') as mock_download:
            
            # Test sidebar_inputs
            with patch('streamlit.sidebar'), \
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
                    
                    # Verify no download button was called
                    mock_download.assert_not_called()

class TestConfigStateVariablesRemoved:
    """Verify config-related session state variables are not used"""
    
    def test_no_config_save_state_variables(self):
        """Verify no config save state variables in session state initialization"""
        
        # Check initialize_session_state function
        with patch.dict('streamlit.session_state', {}, clear=True):
            
            ui.initialize_session_state()
            
            # These config-related variables should NOT be in session state
            forbidden_state_vars = [
                'saved_configurations',
                'current_config_name',
                'config_to_load',
                'uploaded_config_file',
                'config_data_to_load'
            ]
            
            for var in forbidden_state_vars:
                assert var not in st.session_state, \
                    f"Found forbidden config state variable: {var}"

    def test_session_state_only_essential_vars(self):
        """Verify session state only contains essential variables"""
        
        with patch.dict('streamlit.session_state', {}, clear=True):
            
            ui.initialize_session_state()
            
            # Only these variables should be present
            allowed_state_vars = {
                'prediction_history',
                'comparison_results', 
                'form_attempted',
                'prf_size_label2code',
                'prf_size_code2mid',
                'current_shap_values',
                'current_model_explainer',
                'last_prediction_inputs'
            }
            
            actual_vars = set(st.session_state.keys())
            
            # Check for unexpected variables
            unexpected_vars = actual_vars - allowed_state_vars
            assert len(unexpected_vars) == 0, \
                f"Found unexpected session state variables: {unexpected_vars}"

class TestMainFunctionSimplified:
    """Verify main function is simplified without config management"""
    
    def test_main_function_tab_structure(self):
        """Verify main function has correct simplified tab structure"""
        
        with patch('streamlit.title'), \
             patch('streamlit.markdown'), \
             patch('streamlit.tabs') as mock_tabs, \
             patch('streamlit.info'), \
             patch('streamlit.warning'), \
             patch('streamlit.header'), \
             patch('streamlit.expander'), \
             patch.object(ui, 'set_sidebar_width'), \
             patch.object(ui, 'initialize_session_state'), \
             patch.object(ui, 'sidebar_inputs') as mock_sidebar, \
             patch.object(ui, 'display_model_comparison'), \
             patch.object(ui, 'display_static_shap_analysis'), \
             patch.object(ui, 'about_section'):
            
            # Mock sidebar inputs
            mock_sidebar.return_value = {
                'selected_model': 'test_model',
                'submit': False
            }
            
            # Mock tabs
            tabs = [MagicMock() for _ in range(5)]
            mock_tabs.return_value = tabs
            
            # Call main function
            ui.main()
            
            # Verify 5 tabs are created (no config management tabs)
            mock_tabs.assert_called_once()
            tab_call_args = mock_tabs.call_args[0][0]
            
            expected_tabs = [
                "🔮 Estimator", 
                "📊 Visualisations & Analysis", 
                "🤖 Model Comparison", 
                "📈 Static SHAP Analysis", 
                "❓ Help"
            ]
            
            assert tab_call_args == expected_tabs

    def test_main_function_no_config_logic(self):
        """Verify main function contains no config management logic"""
        
        # Get source code of main function
        main_source = inspect.getsource(ui.main)
        
        # Check for absence of config-related logic
        forbidden_config_logic = [
            'save_config',
            'load_config',
            'config_name',
            'uploaded_file',
            'configuration',
            'file_uploader',
            'download_button'
        ]
        
        for forbidden in forbidden_config_logic:
            assert forbidden not in main_source, \
                f"Found forbidden config logic '{forbidden}' in main function"

class TestRemovedImports:
    """Verify config-related imports are removed"""
    
    def test_no_json_import_for_config(self):
        """Verify JSON import is only used for necessary functions"""
        
        # JSON should still be imported for model display names, etc.
        # but not for config save/load
        ui_source = inspect.getsource(ui)
        
        # Should not have config-specific JSON usage
        forbidden_json_usage = [
            'json.dump',
            'json.dumps',
            'json.load', 
            'json.loads'
        ]
        
        # Count occurrences - should be minimal and not for config
        for usage in forbidden_json_usage:
            if usage in ui_source:
                # If found, should not be in config context
                lines_with_usage = [line for line in ui_source.split('\n') if usage in line]
                for line in lines_with_usage:
                    assert 'config' not in line.lower(), \
                        f"Found config-related JSON usage: {line.strip()}"

    def test_no_yaml_import_for_config_save(self):
        """Verify YAML import is not used for config save operations"""
        
        ui_source = inspect.getsource(ui)
        
        # YAML should be used for loading existing configs, not saving user configs
        if 'yaml.dump' in ui_source:
            lines_with_dump = [line for line in ui_source.split('\n') if 'yaml.dump' in line]
            for line in lines_with_dump:
                assert 'user' not in line.lower() and 'save' not in line.lower(), \
                    f"Found user config save with YAML: {line.strip()}"

class TestFunctionSignaturesSimplified:
    """Verify function signatures are simplified"""
    
    def test_sidebar_inputs_simplified_signature(self):
        """Verify sidebar_inputs has simplified signature"""
        
        sig = inspect.signature(ui.sidebar_inputs)
        
        # Should have no config-related parameters
        forbidden_params = [
            'config_name',
            'save_config',
            'load_config'
        ]
        
        actual_params = list(sig.parameters.keys())
        
        for forbidden in forbidden_params:
            assert forbidden not in actual_params, \
                f"Found forbidden parameter '{forbidden}' in sidebar_inputs"

    def test_no_config_helper_functions(self):
        """Verify no config helper functions exist"""
        
        # Get all functions in ui module
        ui_functions = [name for name in dir(ui) if callable(getattr(ui, name))]
        
        # Check for config-related helper functions
        forbidden_function_patterns = [
            'make_config',
            'save_config', 
            'load_config',
            'export_config',
            'import_config',
            'config_to_json',
            'json_to_config'
        ]
        
        for pattern in forbidden_function_patterns:
            matching_functions = [f for f in ui_functions if pattern in f.lower()]
            assert len(matching_functions) == 0, \
                f"Found forbidden config functions: {matching_functions}"

class TestUIFlowSimplified:
    """Verify UI flow is simplified without config steps"""
    
    def test_estimator_tab_direct_flow(self):
        """Verify estimator tab has direct prediction flow"""
        
        with patch.object(ui, 'display_inputs'), \
             patch.object(ui, 'predict_man_hours') as mock_predict, \
             patch.object(ui, 'show_prediction'), \
             patch.object(ui, 'add_prediction_to_history'), \
             patch.object(ui, 'show_prediction_history'), \
             patch.object(ui, 'show_feature_importance'):
            
            mock_predict.return_value = 480.0
            
            # Simulate direct prediction flow
            user_inputs = {
                'project_prf_functional_size': 100,
                'selected_model': 'test_model',
                'submit': True
            }
            
            # Flow should go directly from inputs to prediction
            if user_inputs.get('submit') and user_inputs.get('selected_model'):
                prediction = mock_predict(user_inputs, user_inputs['selected_model'])
                
                # Should predict directly without config save/load steps
                mock_predict.assert_called_once()

    def test_no_config_workflow_interruptions(self):
        """Verify no config workflow interruptions in main flow"""
        
        # Main workflow should be:
        # 1. Fill inputs -> 2. Predict -> 3. Show results -> 4. Analyze
        # No config save/load steps should interrupt this flow
        
        main_source = inspect.getsource(ui.main)
        
        # Should not have workflow interruptions
        forbidden_interruptions = [
            'save_before_predict',
            'load_before_predict',
            'config_required',
            'must_save_config'
        ]
        
        for interruption in forbidden_interruptions:
            assert interruption not in main_source, \
                f"Found workflow interruption: {interruption}"

class TestErrorHandlingSimplified:
    """Verify error handling is simplified without config errors"""
    
    def test_no_config_error_handling(self):
        """Verify no config-specific error handling"""
        
        ui_source = inspect.getsource(ui)
        
        # Should not have config-specific error messages
        forbidden_error_patterns = [
            'config.*not.*found',
            'config.*invalid',
            'save.*failed',
            'load.*failed',
            'upload.*error'
        ]
        
        for pattern in forbidden_error_patterns:
            import re
            if re.search(pattern, ui_source, re.IGNORECASE):
                # Get the specific line for better error message
                lines = ui_source.split('\n')
                matching_lines = [line for line in lines if re.search(pattern, line, re.IGNORECASE)]
                assert False, f"Found forbidden config error handling: {matching_lines[0].strip()}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])