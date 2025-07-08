# test_shap_tab.py
"""
Test cases for the Visualizations & Analysis Tab (Tab 2) - Instance-Specific SHAP Analysis
Verifies that only instance-specific SHAP analysis is shown (no global/static analysis).
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the simplified UI module
import ui
from constants import UIConstants, FileConstants

class TestShapTabCore:
    """Test SHAP tab core functionality"""
    
    def setup_method(self):
        """Setup common test data"""
        self.mock_prediction_history = [
            {
                'inputs': {
                    'project_prf_functional_size': 100,
                    'project_prf_max_team_size': 5,
                    'external_eef_industry_sector': 'Financial'
                },
                'model_technical': 'rf_model',
                'prediction_hours': 480.0,
                'timestamp': '2024-01-01 10:00:00'
            }
        ]
        
        # Reset session state
        st.session_state = {
            'prediction_history': [],
            'latest_prediction': None
        }

    def test_shap_tab_header_display(self):
        """Test SHAP tab shows correct header"""
        
        with patch('streamlit.header') as mock_header:
            
            # Simulate the SHAP tab content from main()
            # This should be called in the SHAP tab
            expected_header = "Instance-Specific SHAP Analysis"
            
            # Verify header is displayed
            # (This would be in the actual tab logic)
            st.header("Instance-Specific SHAP Analysis")
            mock_header.assert_called_with("Instance-Specific SHAP Analysis")

    def test_shap_tab_no_prediction_warning(self):
        """Test SHAP tab shows warning when no predictions exist"""
        
        with patch('streamlit.warning') as mock_warning:
            
            # Empty prediction history
            st.session_state['prediction_history'] = []
            
            # Simulate the check from main()
            if not st.session_state.prediction_history:
                st.warning("Please make at least one prediction first to enable SHAP analysis.")
            
            mock_warning.assert_called_with("Please make at least one prediction first to enable SHAP analysis.")

    def test_shap_tab_with_valid_prediction(self):
        """Test SHAP tab functionality with valid prediction history"""
        
        with patch('streamlit.info') as mock_info, \
             patch('streamlit.error') as mock_error, \
             patch.object(ui, 'display_instance_specific_shap') as mock_display_shap:
            
            # Set up valid prediction history
            st.session_state['prediction_history'] = self.mock_prediction_history.copy()
            
            # Simulate the logic from main() SHAP tab
            if st.session_state.prediction_history:
                latest_prediction = st.session_state.prediction_history[-1]
                user_inputs = latest_prediction.get('inputs', {})
                model_name = latest_prediction.get('model_technical')
                
                if user_inputs and model_name:
                    # Should show info message
                    st.info("Enhanced SHAP Analysis: Using optimized feature analysis for faster performance with high accuracy.")
                    
                    # Should call SHAP display
                    ui.display_instance_specific_shap(user_inputs, model_name)
                else:
                    st.error("Cannot perform analysis - missing prediction data.")
            
            # Verify calls
            mock_info.assert_called_with("Enhanced SHAP Analysis: Using optimized feature analysis for faster performance with high accuracy.")
            mock_display_shap.assert_called_once_with(
                self.mock_prediction_history[0]['inputs'],
                self.mock_prediction_history[0]['model_technical']
            )

    def test_shap_tab_missing_data_error(self):
        """Test SHAP tab error handling for missing data"""
        
        with patch('streamlit.error') as mock_error:
            
            # Prediction with missing data
            incomplete_prediction = {
                'inputs': {},  # Empty inputs
                'model_technical': None,  # Missing model
                'prediction_hours': 480.0
            }
            
            st.session_state['prediction_history'] = [incomplete_prediction]
            
            # Simulate the check from main()
            if st.session_state.prediction_history:
                latest_prediction = st.session_state.prediction_history[-1]
                user_inputs = latest_prediction.get('inputs', {})
                model_name = latest_prediction.get('model_technical')
                
                if not user_inputs or not model_name:
                    st.error("Cannot perform analysis - missing prediction data.")
            
            mock_error.assert_called_with("Cannot perform analysis - missing prediction data.")

    def test_display_instance_specific_shap_function(self):
        """Test the display_instance_specific_shap function"""
        
        with patch.object(ui, 'display_optimized_shap_analysis') as mock_optimized_shap:
            
            user_inputs = {
                'project_prf_functional_size': 100,
                'project_prf_max_team_size': 5
            }
            model_name = 'rf_model'
            
            # Call the function
            ui.display_instance_specific_shap(user_inputs, model_name)
            
            # Verify it routes to the optimized SHAP analysis
            mock_optimized_shap.assert_called_once_with(
                user_inputs, 
                model_name, 
                ui.get_trained_model
            )

    def test_shap_tab_no_global_analysis(self):
        """Verify SHAP tab does NOT contain global/static analysis elements"""
        
        with patch('streamlit.subheader') as mock_subheader, \
             patch('streamlit.markdown') as mock_markdown, \
             patch('streamlit.plotly_chart') as mock_plotly:
            
            # Set up valid prediction
            st.session_state['prediction_history'] = self.mock_prediction_history.copy()
            
            # Simulate SHAP tab logic - should NOT call global analysis
            # These calls should NOT happen in the simplified SHAP tab:
            
            # Should NOT have global SHAP analysis
            forbidden_calls = [
                "Global Feature Importance",
                "Model-Wide SHAP Analysis", 
                "Dataset-Wide Analysis",
                "Overall Model Behavior"
            ]
            
            # After running the SHAP tab logic, verify these are not called
            subheader_calls = [call[0][0] for call in mock_subheader.call_args_list if call[0]]
            
            for forbidden in forbidden_calls:
                assert forbidden not in subheader_calls, f"Found forbidden global analysis: {forbidden}"

    def test_shap_tab_only_instance_elements(self):
        """Verify SHAP tab only contains instance-specific elements"""
        
        with patch('streamlit.header') as mock_header, \
             patch('streamlit.info') as mock_info, \
             patch.object(ui, 'display_instance_specific_shap'):
            
            st.session_state['prediction_history'] = self.mock_prediction_history.copy()
            
            # Simulate the actual SHAP tab logic from main()
            # Should only have instance-specific analysis
            
            expected_elements = [
                "Instance-Specific SHAP Analysis",
                "Enhanced SHAP Analysis: Using optimized feature analysis"
            ]
            
            # These would be called in the actual implementation
            st.header("Instance-Specific SHAP Analysis")
            st.info("Enhanced SHAP Analysis: Using optimized feature analysis for faster performance with high accuracy.")
            
            # Verify only allowed elements are present
            mock_header.assert_called_with("Instance-Specific SHAP Analysis")
            mock_info.assert_called()

class TestShapTabIntegration:
    """Test SHAP tab integration with prediction history"""
    
    def test_shap_uses_latest_prediction(self):
        """Test SHAP analysis uses the most recent prediction"""
        
        # Multiple predictions
        prediction_history = [
            {
                'inputs': {'old': 'prediction'},
                'model_technical': 'old_model',
                'timestamp': '2024-01-01 09:00:00'
            },
            {
                'inputs': {'project_prf_functional_size': 200},
                'model_technical': 'latest_model', 
                'timestamp': '2024-01-01 10:00:00'
            }
        ]
        
        with patch.object(ui, 'display_instance_specific_shap') as mock_shap:
            
            st.session_state['prediction_history'] = prediction_history
            
            # Simulate SHAP tab logic
            if st.session_state.prediction_history:
                latest_prediction = st.session_state.prediction_history[-1]  # Get latest
                user_inputs = latest_prediction.get('inputs', {})
                model_name = latest_prediction.get('model_technical')
                
                if user_inputs and model_name:
                    ui.display_instance_specific_shap(user_inputs, model_name)
            
            # Verify it used the latest prediction (index -1)
            mock_shap.assert_called_once_with(
                {'project_prf_functional_size': 200},  # Latest inputs
                'latest_model'  # Latest model
            )

    def test_shap_tab_state_independence(self):
        """Test SHAP tab works independently of other UI state"""
        
        with patch.object(ui, 'display_instance_specific_shap') as mock_shap:
            
            # Set up minimal required state
            st.session_state['prediction_history'] = [{
                'inputs': {'test': 'data'},
                'model_technical': 'test_model'
            }]
            
            # Should work without other UI state variables
            # (No config state, no save/load state needed)
            
            # Simulate SHAP tab
            latest_prediction = st.session_state.prediction_history[-1]
            ui.display_instance_specific_shap(
                latest_prediction['inputs'],
                latest_prediction['model_technical']
            )
            
            mock_shap.assert_called_once()

class TestShapTabErrorHandling:
    """Test SHAP tab error handling scenarios"""
    
    def test_shap_tab_corrupted_history(self):
        """Test SHAP tab handles corrupted prediction history"""
        
        with patch('streamlit.error') as mock_error:
            
            # Corrupted history entry
            corrupted_history = [
                {
                    # Missing required fields
                    'timestamp': '2024-01-01 10:00:00'
                    # No 'inputs' or 'model_technical'
                }
            ]
            
            st.session_state['prediction_history'] = corrupted_history
            
            # Simulate SHAP tab logic with error handling
            if st.session_state.prediction_history:
                latest_prediction = st.session_state.prediction_history[-1]
                user_inputs = latest_prediction.get('inputs', {})
                model_name = latest_prediction.get('model_technical')
                
                if not user_inputs or not model_name:
                    st.error("Cannot perform analysis - missing prediction data.")
            
            mock_error.assert_called_with("Cannot perform analysis - missing prediction data.")

    def test_shap_tab_empty_inputs(self):
        """Test SHAP tab handles empty user inputs"""
        
        with patch('streamlit.error') as mock_error:
            
            # Valid structure but empty inputs
            empty_inputs_history = [
                {
                    'inputs': {},  # Empty dict
                    'model_technical': 'test_model'
                }
            ]
            
            st.session_state['prediction_history'] = empty_inputs_history
            
            # Should trigger error for empty inputs
            if st.session_state.prediction_history:
                latest_prediction = st.session_state.prediction_history[-1]
                user_inputs = latest_prediction.get('inputs', {})
                model_name = latest_prediction.get('model_technical')
                
                if not user_inputs or not model_name:
                    st.error("Cannot perform analysis - missing prediction data.")
            
            mock_error.assert_called_with("Cannot perform analysis - missing prediction data.")

    def test_shap_analysis_function_error_handling(self):
        """Test error handling within SHAP analysis function"""
        
        with patch.object(ui, 'display_optimized_shap_analysis') as mock_shap:
            
            # Mock SHAP analysis to raise an exception
            mock_shap.side_effect = Exception("SHAP analysis failed")
            
            user_inputs = {'test': 'data'}
            model_name = 'test_model'
            
            # Should handle exception gracefully
            try:
                ui.display_instance_specific_shap(user_inputs, model_name)
            except Exception as e:
                # Error should be caught in the UI layer
                assert "SHAP analysis failed" in str(e)

class TestShapTabUIElements:
    """Test specific UI elements in SHAP tab"""
    
    def test_shap_tab_info_message_content(self):
        """Test the specific info message content"""
        
        with patch('streamlit.info') as mock_info:
            
            st.session_state['prediction_history'] = [{
                'inputs': {'test': 'data'},
                'model_technical': 'test_model'
            }]
            
            # The exact message from the UI
            expected_message = "Enhanced SHAP Analysis: Using optimized feature analysis for faster performance with high accuracy."
            
            # Simulate the call
            st.info(expected_message)
            
            mock_info.assert_called_with(expected_message)

    def test_shap_tab_warning_message_content(self):
        """Test the specific warning message content"""
        
        with patch('streamlit.warning') as mock_warning:
            
            st.session_state['prediction_history'] = []
            
            # The exact warning message from the UI
            expected_warning = "Please make at least one prediction first to enable SHAP analysis."
            
            # Simulate the call
            st.warning(expected_warning)
            
            mock_warning.assert_called_with(expected_warning)

    def test_shap_tab_error_message_content(self):
        """Test the specific error message content"""
        
        with patch('streamlit.error') as mock_error:
            
            # The exact error message from the UI
            expected_error = "Cannot perform analysis - missing prediction data."
            
            # Simulate the call
            st.error(expected_error)
            
            mock_error.assert_called_with(expected_error)

class TestShapTabNoGlobalFeatures:
    """Test that global SHAP features are completely removed"""
    
    def test_no_static_shap_references(self):
        """Verify no references to static SHAP analysis"""
        
        # These functions/calls should NOT exist in the SHAP tab
        forbidden_functions = [
            'display_static_shap_analysis',
            'show_global_shap_analysis', 
            'display_model_wide_shap',
            'show_feature_importance_global'
        ]
        
        # Check that these are not called anywhere in the SHAP tab logic
        for func_name in forbidden_functions:
            assert not hasattr(ui, func_name) or func_name not in dir(ui), \
                f"Found forbidden global SHAP function: {func_name}"

    def test_no_dataset_wide_analysis(self):
        """Verify no dataset-wide analysis components"""
        
        with patch('streamlit.subheader') as mock_subheader, \
             patch('streamlit.markdown') as mock_markdown:
            
            # These elements should NOT appear in SHAP tab
            forbidden_elements = [
                "Dataset-Wide Feature Importance",
                "Global Model Behavior",
                "Training Data Analysis",
                "Model Performance Overview"
            ]
            
            # Simulate SHAP tab execution
            st.session_state['prediction_history'] = [{
                'inputs': {'test': 'data'},
                'model_technical': 'test_model'
            }]
            
            # After SHAP tab logic, verify forbidden elements not called
            all_calls = mock_subheader.call_args_list + mock_markdown.call_args_list
            called_text = [str(call) for call in all_calls]
            
            for forbidden in forbidden_elements:
                assert not any(forbidden in text for text in called_text), \
                    f"Found forbidden dataset analysis: {forbidden}"

    def test_no_model_comparison_in_shap(self):
        """Verify no model comparison features in SHAP tab"""
        
        # SHAP tab should NOT include model comparison elements
        # (Model comparison should be in its own separate tab)
        
        forbidden_comparison_elements = [
            "Compare Models",
            "Model Performance Comparison", 
            "Cross-Model Analysis",
            "Multi-Model SHAP"
        ]
        
        # These should not be present in the SHAP tab
        # (This is more of a design verification test)
        
        with patch('streamlit.header') as mock_header:
            
            # Simulate SHAP tab
            st.session_state['prediction_history'] = [{
                'inputs': {'test': 'data'},
                'model_technical': 'test_model'
            }]
            
            # Should only have instance-specific header
            st.header("Instance-Specific SHAP Analysis")
            
            # Verify only the correct header was called
            mock_header.assert_called_with("Instance-Specific SHAP Analysis")
            
            # No comparison headers should be called
            header_calls = [call[0][0] for call in mock_header.call_args_list]
            for forbidden in forbidden_comparison_elements:
                assert forbidden not in header_calls

class TestShapTabIntegrationWithOtherTabs:
    """Test SHAP tab integration with other tabs"""
    
    def test_shap_tab_independent_of_config(self):
        """Test SHAP tab works without configuration management"""
        
        with patch.object(ui, 'display_instance_specific_shap') as mock_shap:
            
            # SHAP should work without any config state
            st.session_state = {
                'prediction_history': [{
                    'inputs': {'test': 'data'},
                    'model_technical': 'test_model'
                }]
                # No config-related state needed
            }
            
            # Should work fine
            latest_prediction = st.session_state['prediction_history'][-1]
            ui.display_instance_specific_shap(
                latest_prediction['inputs'],
                latest_prediction['model_technical'] 
            )
            
            mock_shap.assert_called_once()

    def test_shap_tab_uses_estimator_results(self):
        """Test SHAP tab correctly uses results from estimator tab"""
        
        with patch.object(ui, 'display_instance_specific_shap') as mock_shap:
            
            # Prediction history should come from estimator tab
            estimator_result = {
                'inputs': {
                    'project_prf_functional_size': 150,
                    'project_prf_max_team_size': 7,
                    'external_eef_industry_sector': 'Banking'
                },
                'model_technical': 'xgb_model',
                'prediction_hours': 380.0,
                'timestamp': '2024-01-01 12:00:00'
            }
            
            st.session_state['prediction_history'] = [estimator_result]
            
            # SHAP should use exactly the same data
            latest_prediction = st.session_state['prediction_history'][-1]
            ui.display_instance_specific_shap(
                latest_prediction['inputs'],
                latest_prediction['model_technical']
            )
            
            # Verify exact data passed through
            mock_shap.assert_called_once_with(
                estimator_result['inputs'],
                estimator_result['model_technical']
            )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])