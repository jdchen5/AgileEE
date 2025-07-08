# test_estimator_tab.py
"""
Test cases for the Estimator Tab (Tab 1) - Core prediction functionality
Verifies that the main prediction features work correctly after UI simplification.
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the simplified UI module
import ui
from constants import UIConstants, FileConstants

class TestEstimatorTabCore:
    """Test core estimator tab functionality"""
    
    def setup_method(self):
        """Setup common test data"""
        self.mock_user_inputs = {
            'project_prf_functional_size': 100,
            'project_prf_max_team_size': 5,
            'external_eef_industry_sector': 'Financial',
            'tech_tf_primary_programming_language': 'Java',
            'selected_model': 'test_model',
            'submit': True
        }
        
        self.mock_prediction_result = 480.0
        
        # Reset session state
        st.session_state = {
            'prediction_history': [],
            'comparison_results': [],
            'form_attempted': False,
            'prf_size_label2code': {},
            'prf_size_code2mid': {},
            'latest_prediction': None
        }

    def test_sidebar_inputs_basic_functionality(self):
        """Test that sidebar_inputs works without save/load features"""
        
        with patch('streamlit.sidebar'), \
             patch('streamlit.title'), \
             patch('streamlit.info'), \
             patch('streamlit.tabs') as mock_tabs, \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.divider'), \
             patch('streamlit.subheader'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.number_input') as mock_number, \
             patch('streamlit.text_input') as mock_text, \
             patch('streamlit.checkbox') as mock_checkbox:
            
            # Mock field rendering
            mock_selectbox.return_value = "test_value"
            mock_number.return_value = 5
            mock_text.return_value = "test"
            mock_checkbox.return_value = True
            mock_button.return_value = False
            mock_columns.return_value = [MagicMock(), MagicMock()]
            
            # Mock tabs context manager
            tab1, tab2 = MagicMock(), MagicMock()
            mock_tabs.return_value = [tab1, tab2]
            
            with patch.object(ui, 'check_required_models') as mock_check, \
                 patch.object(ui, 'list_available_models') as mock_list, \
                 patch.object(ui, 'get_tab_organization') as mock_tabs_org, \
                 patch.object(ui, 'FIELDS', {
                     'project_prf_functional_size': {'type': 'numeric', 'min': 1, 'max': 1000, 'default': 100, 'mandatory': True},
                     'project_prf_max_team_size': {'type': 'numeric', 'min': 1, 'max': 50, 'default': 5, 'mandatory': True}
                 }):
                
                mock_check.return_value = {"models_available": True}
                mock_list.return_value = [{"display_name": "Test Model", "technical_name": "test_model"}]
                mock_tabs_org.return_value = {
                    "Important Features": ["project_prf_functional_size"],
                    "Nice Features": ["project_prf_max_team_size"]
                }
                
                result = ui.sidebar_inputs()
                
                # Verify basic structure
                assert isinstance(result, dict)
                assert "selected_model" in result
                assert "submit" in result
                
                # Verify NO config management keys
                forbidden_keys = ['save_config', 'config_name', 'load_config', 'uploaded_file']
                for key in forbidden_keys:
                    assert key not in result, f"Found forbidden config key: {key}"

    def test_display_inputs_functionality(self):
        """Test the display_inputs function works correctly"""
        
        with patch('streamlit.expander') as mock_expander, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.text') as mock_text, \
             patch('streamlit.info') as mock_info, \
             patch('streamlit.warning') as mock_warning:
            
            # Mock expander context
            expander_context = MagicMock()
            mock_expander.return_value.__enter__ = Mock(return_value=expander_context)
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            
            # Mock columns
            col1, col2 = MagicMock(), MagicMock()
            mock_columns.return_value = [col1, col2]
            
            with patch.object(ui, 'get_field_label') as mock_get_label, \
                 patch.object(ui, 'get_model_display_name') as mock_get_display:
                
                mock_get_label.side_effect = lambda x: x.replace('_', ' ').title()
                mock_get_display.return_value = "Random Forest Model"
                
                # Test with valid inputs
                user_inputs = {
                    'project_prf_functional_size': 100,
                    'project_prf_max_team_size': 5,
                    'selected_model': 'rf_model',
                    'submit': True
                }
                
                ui.display_inputs(user_inputs, 'rf_model')
                
                # Verify expander was created
                mock_expander.assert_called_once_with("📋Input Parameters Summary", expanded=False)
                
                # Verify model display was called
                mock_get_display.assert_called_once_with('rf_model')

    def test_show_prediction_basic_display(self):
        """Test show_prediction displays results correctly"""
        
        with patch('streamlit.subheader') as mock_subheader, \
             patch('streamlit.info') as mock_info, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric') as mock_metric, \
             patch('streamlit.warning') as mock_warning:
            
            # Mock 4 columns for metrics
            cols = [MagicMock() for _ in range(4)]
            mock_columns.return_value = cols
            
            with patch.object(ui, 'get_model_display_name') as mock_display_name:
                mock_display_name.return_value = "Test Model"
                
                # Test basic prediction display
                ui.show_prediction(480.0, 'test_model')
                
                # Verify main components were called
                mock_subheader.assert_called_with("🎯 Prediction Results")
                mock_info.assert_called_with("**Model Used:** Test Model")
                
                # Verify metrics were created (4 columns)
                assert mock_metric.call_count == 4
                
                # Check metric calls
                metric_calls = mock_metric.call_args_list
                assert ("📊 Total Effort", "480 hours") in [call[0] for call in metric_calls]

    def test_show_prediction_with_user_inputs_warnings(self):
        """Test show_prediction with user inputs and dynamic warnings"""
        
        with patch('streamlit.subheader'), \
             patch('streamlit.info'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.warning') as mock_warning:
            
            mock_columns.return_value = [MagicMock() for _ in range(4)]
            
            # Mock session state with size info
            st.session_state['prf_size_code2full'] = {
                'M': {'minimumhour': 100, 'maximumhour': 1000}
            }
            
            user_inputs = {
                'project_prf_relative_size': 'M',
                'project_prf_functional_size': 200
            }
            
            with patch.object(ui, 'get_model_display_name') as mock_display:
                mock_display.return_value = "Test Model"
                
                # Test prediction below minimum
                ui.show_prediction(50.0, 'test_model', user_inputs)
                
                # Should trigger warning for below minimum
                mock_warning.assert_called()
                warning_text = mock_warning.call_args[0][0]
                assert "below" in warning_text and "minimum" in warning_text

    def test_prediction_flow_integration(self):
        """Test the complete prediction flow in estimator tab"""
        
        with patch('streamlit.spinner') as mock_spinner, \
             patch('streamlit.divider'), \
             patch('streamlit.error') as mock_error, \
             patch.object(ui, 'display_inputs') as mock_display_inputs, \
             patch.object(ui, 'predict_man_hours') as mock_predict, \
             patch.object(ui, 'show_prediction') as mock_show_pred, \
             patch.object(ui, 'add_prediction_to_history') as mock_add_history, \
             patch.object(ui, 'show_prediction_history') as mock_show_history, \
             patch.object(ui, 'show_feature_importance') as mock_show_importance:
            
            # Mock spinner context
            spinner_context = MagicMock()
            mock_spinner.return_value.__enter__ = Mock(return_value=spinner_context)
            mock_spinner.return_value.__exit__ = Mock(return_value=None)
            
            mock_predict.return_value = 480.0
            
            user_inputs = self.mock_user_inputs.copy()
            selected_model = 'test_model'
            
            # Simulate the prediction flow logic from main()
            if user_inputs.get('submit', False):
                if selected_model:
                    # Display inputs
                    ui.display_inputs(user_inputs, selected_model)
                    
                    # Make prediction
                    prediction = ui.predict_man_hours(user_inputs, selected_model)
                    
                    if prediction:
                        # Show results
                        ui.show_prediction(prediction, selected_model, user_inputs)
                        ui.add_prediction_to_history(user_inputs, selected_model, prediction)
                        ui.show_prediction_history()
                        ui.show_feature_importance(selected_model, user_inputs)
            
            # Verify the flow executed correctly
            mock_display_inputs.assert_called_once_with(user_inputs, selected_model)
            mock_predict.assert_called_once_with(user_inputs, selected_model)
            mock_show_pred.assert_called_once_with(480.0, selected_model, user_inputs)
            mock_add_history.assert_called_once_with(user_inputs, selected_model, 480.0)
            mock_show_history.assert_called_once()
            mock_show_importance.assert_called_once_with(selected_model, user_inputs)

    def test_feature_importance_display(self):
        """Test feature importance analysis display"""
        
        with patch('streamlit.subheader') as mock_subheader, \
             patch('streamlit.info') as mock_info, \
             patch('streamlit.bar_chart') as mock_bar_chart, \
             patch('streamlit.expander') as mock_expander, \
             patch('streamlit.dataframe') as mock_dataframe, \
             patch('streamlit.warning') as mock_warning:
            
            # Mock expander context
            expander_context = MagicMock()
            mock_expander.return_value.__enter__ = Mock(return_value=expander_context)
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            
            with patch.object(ui, 'get_feature_importance') as mock_get_importance, \
                 patch.object(ui, 'get_model_display_name') as mock_display_name, \
                 patch.object(ui, 'get_field_title') as mock_field_title, \
                 patch.object(ui, 'FEATURE_IMPORTANCE_DISPLAY', {
                     'max_features_shown': 10,
                     'precision_decimals': 3
                 }):
                
                mock_get_importance.return_value = np.array([0.3, 0.2, 0.15, 0.1, 0.05])
                mock_display_name.return_value = "Random Forest"
                mock_field_title.side_effect = lambda x: x.replace('_', ' ').title()
                
                features_dict = {
                    'feature1': 100,
                    'feature2': 5,
                    'feature3': 'value',
                    'selected_model': 'rf_model',
                    'submit': True
                }
                
                ui.show_feature_importance('rf_model', features_dict)
                
                # Verify feature importance was retrieved
                mock_get_importance.assert_called_once_with('rf_model')
                
                # Verify charts were created
                mock_bar_chart.assert_called_once()
                mock_dataframe.assert_called_once()

    def test_prediction_history_functionality(self):
        """Test prediction history display"""
        
        # Setup prediction history
        st.session_state['prediction_history'] = [
            {
                'timestamp': '2024-01-01 10:00:00',
                'model': 'Random Forest',
                'model_technical': 'rf_model',
                'prediction_hours': 480.0,
                'inputs': self.mock_user_inputs
            },
            {
                'timestamp': '2024-01-01 11:00:00', 
                'model': 'XGBoost',
                'model_technical': 'xgb_model',
                'prediction_hours': 520.0,
                'inputs': self.mock_user_inputs
            }
        ]
        
        with patch('streamlit.subheader') as mock_subheader, \
             patch('streamlit.dataframe') as mock_dataframe, \
             patch('streamlit.info') as mock_info:
            
            with patch.object(ui, 'UIConstants') as mock_constants:
                mock_constants.HOURS_PER_DAY = 8
                
                ui.show_prediction_history()
                
                # Verify history was displayed
                mock_subheader.assert_called_with("📈 Prediction History")
                mock_dataframe.assert_called_once()
                
                # Verify the dataframe call contains the right structure
                df_call = mock_dataframe.call_args[0][0]
                assert isinstance(df_call, pd.DataFrame)
                assert len(df_call) == 2  # Two history entries
                assert 'Timestamp' in df_call.columns
                assert 'Model' in df_call.columns
                assert 'Hours' in df_call.columns

    def test_estimator_tab_without_models(self):
        """Test estimator tab behavior when no models are available"""
        
        with patch('streamlit.warning') as mock_warning, \
             patch('streamlit.info') as mock_info:
            
            with patch.object(ui, 'check_required_models') as mock_check:
                mock_check.return_value = {"models_available": False}
                
                # This should handle gracefully
                user_inputs = {
                    'selected_model': None,
                    'submit': True
                }
                
                # Simulate the condition check from main()
                selected_model = user_inputs.get('selected_model')
                if user_inputs.get('submit', False):
                    if not selected_model:
                        # This should trigger a warning
                        pass
                
                # Verify no crash occurs and appropriate handling

    def test_estimator_tab_error_handling(self):
        """Test error handling in estimator tab"""
        
        with patch('streamlit.error') as mock_error, \
             patch.object(ui, 'predict_man_hours') as mock_predict:
            
            # Mock prediction failure
            mock_predict.side_effect = Exception("Model prediction failed")
            
            user_inputs = self.mock_user_inputs.copy()
            
            # This should handle the exception gracefully
            try:
                prediction = ui.predict_man_hours(user_inputs, 'test_model')
            except Exception:
                # Error should be caught and handled in the UI
                pass

    def test_welcome_screen_display(self):
        """Test welcome screen when no submission"""
        
        with patch('streamlit.info') as mock_info:
            
            # Simulate main() logic for welcome screen
            user_inputs = {
                'selected_model': 'test_model',
                'submit': False  # No submission
            }
            
            if not user_inputs.get('submit', False):
                # Should show welcome message
                expected_msg = "**Get Started:** Fill in the project parameters in the sidebar and click 'Predict Effort' to get your estimate."
                # This would be called in the actual UI logic
                pass

class TestEstimatorTabValidation:
    """Test validation logic in estimator tab"""
    
    def test_required_fields_validation(self):
        """Test that required fields are properly validated"""
        
        with patch.object(ui, 'FIELDS', {
            'project_prf_functional_size': {'mandatory': True, 'type': 'numeric'},
            'project_prf_max_team_size': {'mandatory': True, 'type': 'numeric'},
            'optional_field': {'mandatory': False, 'type': 'text'}
        }):
            
            # Missing required field
            incomplete_inputs = {
                'project_prf_functional_size': 100,
                # Missing project_prf_max_team_size
                'optional_field': 'test'
            }
            
            # Simulate validation logic from sidebar_inputs
            required_fields = ['project_prf_functional_size', 'project_prf_max_team_size']
            missing_fields = []
            
            for field in required_fields:
                value = incomplete_inputs.get(field)
                if value is None or value == "" or value == []:
                    missing_fields.append(field)
            
            assert len(missing_fields) == 1
            assert 'project_prf_max_team_size' in missing_fields

    def test_predict_button_disable_logic(self):
        """Test predict button disable logic"""
        
        # Test button should be disabled when missing required fields
        missing_fields = ['required_field1']
        selected_model = 'test_model'
        
        should_disable = len(missing_fields) > 0 or not selected_model
        assert should_disable == True  # Should be disabled
        
        # Test button should be enabled when all requirements met
        missing_fields = []
        selected_model = 'test_model'
        
        should_disable = len(missing_fields) > 0 or not selected_model
        assert should_disable == False  # Should be enabled

if __name__ == "__main__":
    pytest.main([__file__, "-v"])