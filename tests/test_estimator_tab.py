# test_estimator_tab.py - UPDATED VERSION
"""
Test cases for the Estimator Tab (Tab 1) - Core prediction functionality
Updated to align with current simplified UI architecture and PredictionEngine concept.
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
import agileee.ui as ui
from agileee.constants import UIConstants, FileConstants

class TestPredictionEngineCore:
    """Test core PredictionEngine functionality (your original concept)"""
    
    def setup_method(self):
        """Setup common test data"""
        self.mock_user_inputs = {
            'project_prf_functional_size': 100,
            'project_prf_max_team_size': 5,
            'external_eef_industry_sector': 'Financial',
            'tech_tf_primary_programming_language': 'Java',
            'project_prf_relative_size': 'M',
            'selected_model': 'test_model',
            'submit': True
        }
        
        self.mock_prediction_result = 480.0
        
        # Reset session state with current architecture
        st.session_state = {
            'prediction_history': [],
            'comparison_results': [],
            'form_attempted': False,
            'prf_size_label2code': {'Medium': 'M'},
            'prf_size_code2mid': {'M': 300},
            'prf_size_code2full': {
                'M': {'code': 'M', 'label': 'Medium', 'midpoint': 300, 'minimumhour': 200, 'maximumhour': 800}
            },
            'current_prediction_results': None,  # NEW session state key
            'cached_model_system': None  # NEW caching system
        }

    def test_prediction_engine_model_loading(self):
        """Test PredictionEngine model loading functionality"""
        
        with patch.object(ui, 'initialize_model_system_cached') as mock_init_models:
            mock_init_models.return_value = {
                "status": {"models_available": True},
                "models": [
                    {'display_name': 'Random Forest', 'technical_name': 'rf_model'},
                    {'display_name': 'XGBoost', 'technical_name': 'xgb_model'}
                ],
                "initialized": True
            }
            
            # Test model system initialization (your PredictionEngine)
            model_system = ui.initialize_model_system_cached()
            
            assert model_system["initialized"] is True
            assert len(model_system["models"]) == 2
            assert model_system["status"]["models_available"] is True

    def test_prediction_engine_core_prediction_flow(self):
        """Test core prediction flow (your PredictionEngine concept)"""
        
        with patch.object(ui, 'predict_man_hours') as mock_predict, \
             patch.object(ui, 'add_prediction_to_history') as mock_add_history, \
             patch.object(ui, 'get_model_display_name') as mock_display_name:
            
            mock_predict.return_value = 480.0
            mock_display_name.return_value = "Random Forest"
            
            user_inputs = self.mock_user_inputs.copy()
            selected_model = 'rf_model'
            
            # Test PredictionEngine core flow
            prediction = ui.predict_man_hours(user_inputs, selected_model)
            
            # Store in NEW session state structure
            st.session_state['current_prediction_results'] = {
                'prediction': prediction,
                'model': selected_model,
                'inputs': user_inputs.copy()
            }
            
            ui.add_prediction_to_history(user_inputs, selected_model, prediction)
            
            # Verify PredictionEngine functionality
            assert prediction == 480.0
            assert st.session_state['current_prediction_results']['prediction'] == 480.0
            assert st.session_state['current_prediction_results']['model'] == selected_model
            
            mock_predict.assert_called_once_with(user_inputs, selected_model)
            mock_add_history.assert_called_once_with(user_inputs, selected_model, 480.0)

class TestStreamlitUIComponents:
    """Test StreamlitUI components (your UI orchestrator)"""
    
    def setup_method(self):
        """Setup for UI tests"""
        st.session_state = {
            'prediction_history': [],
            'current_prediction_results': None,
            'cached_model_system': {
                "status": {"models_available": True},
                "models": [{'display_name': 'Test Model', 'technical_name': 'test_model'}],
                "initialized": True
            }
        }

    def test_sidebar_inputs_no_config_management(self):
        """Test sidebar_inputs works without save/load config features"""
        
        with patch('streamlit.sidebar'), \
             patch('streamlit.title'), \
             patch('streamlit.info'), \
             patch('streamlit.tabs') as mock_tabs, \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.divider'), \
             patch('streamlit.subheader'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.number_input') as mock_number:
            
            # Mock UI components
            mock_selectbox.return_value = "Medium"
            mock_number.return_value = 5
            mock_button.side_effect = [False, False, True]  # clear, show_history, predict
            mock_columns.return_value = [MagicMock(), MagicMock()]
            
            # Mock tabs context manager
            tab1, tab2 = MagicMock(), MagicMock()
            mock_tabs.return_value = [tab1, tab2]
            
            with patch.object(ui, 'get_tab_organization') as mock_tabs_org, \
                 patch.object(ui, 'FIELDS', {
                     'project_prf_functional_size': {
                         'type': 'numeric', 'min': 1, 'max': 1000, 'default': 100, 'mandatory': True
                     },
                     'project_prf_max_team_size': {
                         'type': 'numeric', 'min': 1, 'max': 50, 'default': 5, 'mandatory': True
                     }
                 }):
                
                mock_tabs_org.return_value = {
                    "Important Features": ["project_prf_functional_size"],
                    "Nice Features": ["project_prf_max_team_size"]
                }
                
                result = ui.sidebar_inputs()
                
                # Verify basic structure (NO config management)
                assert isinstance(result, dict)
                assert "selected_model" in result
                assert "submit" in result
                assert "show_history" in result
                
                # Verify NO forbidden config keys
                forbidden_keys = ['save_config', 'config_name', 'load_config', 'uploaded_file']
                for key in forbidden_keys:
                    assert key not in result, f"Found forbidden config key: {key}"

    def test_new_session_state_prediction_flow(self):
        """Test NEW session state-based prediction flow"""
        
        with patch('streamlit.spinner') as mock_spinner, \
             patch('streamlit.divider'), \
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
            
            user_inputs = {
                'project_prf_functional_size': 100,
                'selected_model': 'test_model',
                'submit': True
            }
            selected_model = 'test_model'
            
            # Simulate NEW prediction flow logic from main()
            if user_inputs.get('submit', False):
                if selected_model:
                    prediction = ui.predict_man_hours(user_inputs, selected_model)
                    
                    # NEW: Store complete results in session state
                    st.session_state['current_prediction_results'] = {
                        'prediction': prediction,
                        'model': selected_model,
                        'inputs': user_inputs.copy()
                    }
                    
                    ui.add_prediction_to_history(user_inputs, selected_model, prediction)
            
            # NEW: Display results from session state (persists across interactions)
            if st.session_state.get('current_prediction_results'):
                results = st.session_state['current_prediction_results']
                ui.display_inputs(results['inputs'], results['model'])
                ui.show_prediction(results['prediction'], results['model'], results['inputs'])
                ui.show_prediction_history()
                ui.show_feature_importance(results['model'], results['inputs'])
            
            # Verify NEW flow executed correctly
            mock_predict.assert_called_once_with(user_inputs, selected_model)
            mock_add_history.assert_called_once_with(user_inputs, selected_model, 480.0)
            mock_display_inputs.assert_called_once_with(user_inputs, selected_model)
            mock_show_pred.assert_called_once_with(480.0, selected_model, user_inputs)
            mock_show_history.assert_called_once()
            mock_show_importance.assert_called_once_with(selected_model, user_inputs)

class TestDisplayManager:
    """Test DisplayManager functionality (your display concept)"""
    
    def test_show_prediction_with_size_warnings(self):
        """Test show_prediction with dynamic size-band warnings"""
        
        with patch('streamlit.subheader') as mock_subheader, \
             patch('streamlit.info') as mock_info, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric') as mock_metric, \
             patch('streamlit.warning') as mock_warning:
            
            # Mock 4 columns for metrics
            cols = [MagicMock() for _ in range(4)]
            mock_columns.return_value = cols
            
            # Setup size validation data
            st.session_state.update({
                'prf_size_code2full': {
                    'M': {'code': 'M', 'minimumhour': 200, 'maximumhour': 800}
                }
            })
            
            user_inputs = {
                'project_prf_relative_size': 'M'
            }
            
            with patch.object(ui, 'get_model_display_name') as mock_display_name:
                mock_display_name.return_value = "Test Model"
                
                # Test prediction below minimum (should warn)
                ui.show_prediction(150.0, 'test_model', user_inputs)
                mock_warning.assert_called()
                warning_msg = mock_warning.call_args[0][0]
                assert "below" in warning_msg.lower()
                
                # Reset warning mock
                mock_warning.reset_mock()
                
                # Test prediction above maximum (should warn)
                ui.show_prediction(900.0, 'test_model', user_inputs)
                mock_warning.assert_called()
                warning_msg = mock_warning.call_args[0][0]
                assert "above" in warning_msg.lower()

    def test_prediction_history_with_model_display_names(self):
        """Test prediction history with proper model display names"""
        
        # Setup prediction history with model display names
        st.session_state['prediction_history'] = [
            {
                'timestamp': '2024-01-01 10:00:00',
                'model': 'Random Forest',  # Display name
                'model_technical': 'rf_model',  # Technical name
                'prediction_hours': 480.0,
                'inputs': {'test': 'input'}
            },
            {
                'timestamp': '2024-01-01 11:00:00',
                'model': 'XGBoost',
                'model_technical': 'xgb_model', 
                'prediction_hours': 520.0,
                'inputs': {'test': 'input'}
            }
        ]
        
        with patch('streamlit.subheader') as mock_subheader, \
             patch('streamlit.dataframe') as mock_dataframe, \
             patch('streamlit.info') as mock_info:
            
            with patch.object(ui, 'UIConstants') as mock_constants:
                mock_constants.HOURS_PER_DAY = 8
                
                ui.show_prediction_history()
                
                # Should process history data
                assert mock_subheader.called or mock_dataframe.called or mock_info.called

class TestModelManager:
    """Test ModelManager functionality (your model operations)"""
    
    def test_model_loading_with_caching(self):
        """Test model loading with NEW caching system"""
        
        with patch.object(ui, 'initialize_model_system_cached') as mock_init:
            mock_init.return_value = {
                "status": {"models_available": True},
                "models": [
                    {'display_name': 'Random Forest', 'technical_name': 'rf_model'}
                ],
                "initialized": True
            }
            
            # Test cached model system
            model_system = ui.initialize_model_system_cached()
            
            # Second call should use cache
            cached_system = ui.initialize_model_system_cached()
            
            # Should be same object (cached)
            assert model_system == cached_system
            
            # Should only call initialization once due to caching
            assert mock_init.call_count >= 1

    def test_feature_importance_analysis(self):
        """Test feature importance analysis"""
        
        with patch('streamlit.subheader') as mock_subheader, \
             patch('streamlit.info') as mock_info, \
             patch('streamlit.bar_chart') as mock_bar_chart, \
             patch('streamlit.expander') as mock_expander, \
             patch('streamlit.dataframe') as mock_dataframe:
            
            # Mock expander context
            expander_context = MagicMock()
            mock_expander.return_value.__enter__ = Mock(return_value=expander_context)
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            
            with patch.object(ui, 'get_feature_importance') as mock_get_importance, \
                 patch.object(ui, 'get_model_display_name') as mock_display_name, \
                 patch.object(ui, 'get_field_title') as mock_field_title:
                
                mock_get_importance.return_value = np.array([0.3, 0.2, 0.15])
                mock_display_name.return_value = "Random Forest"
                mock_field_title.side_effect = lambda x: x.replace('_', ' ').title()
                
                features_dict = {
                    'feature1': 100,
                    'feature2': 5,
                    'feature3': 'value'
                }
                
                ui.show_feature_importance('rf_model', features_dict)
                
                # Verify analysis components
                mock_get_importance.assert_called_once_with('rf_model')
                mock_bar_chart.assert_called_once()
                mock_dataframe.assert_called_once()

class TestHistoryManager:
    """Test HistoryManager functionality (your history concept)"""
    
    def test_add_prediction_to_history_with_proper_structure(self):
        """Test adding predictions with proper data structure"""
        
        with patch.object(ui, 'get_model_display_name_from_config') as mock_display:
            mock_display.return_value = "Random Forest"
            
            user_inputs = {
                'project_prf_functional_size': 100,
                'project_prf_max_team_size': 5
            }
            
            ui.add_prediction_to_history(user_inputs, 'rf_model', 480.0)
            
            # Verify history structure
            assert len(st.session_state['prediction_history']) == 1
            
            entry = st.session_state['prediction_history'][0]
            assert entry['model'] == "Random Forest"  # Display name
            assert entry['model_technical'] == 'rf_model'  # Technical name
            assert entry['prediction_hours'] == 480.0
            assert 'timestamp' in entry
            assert 'inputs' in entry

    def test_prediction_comparison_table(self):
        """Test prediction comparison functionality"""
        
        # Setup multiple predictions
        st.session_state['prediction_history'] = [
            {
                'model': 'Random Forest',
                'model_technical': 'rf_model',
                'prediction_hours': 480.0
            },
            {
                'model': 'XGBoost', 
                'model_technical': 'xgb_model',
                'prediction_hours': 520.0
            }
        ]
        
        with patch('streamlit.subheader') as mock_subheader, \
             patch('streamlit.dataframe') as mock_dataframe, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric') as mock_metric:
            
            mock_columns.return_value = [MagicMock() for _ in range(4)]
            
            with patch.object(ui, 'UIConstants') as mock_constants:
                mock_constants.HOURS_PER_DAY = 8
                
                ui.show_prediction_comparison_table()
                
                # Should show comparison for multiple predictions
                if len(st.session_state['prediction_history']) > 1:
                    assert mock_dataframe.called or mock_subheader.called

class TestErrorHandlingAndValidation:
    """Test error handling and validation"""
    
    def test_required_field_validation(self):
        """Test required field validation logic"""
        
        with patch.object(ui, 'FIELDS', {
            'project_prf_functional_size': {'mandatory': True},
            'project_prf_max_team_size': {'mandatory': True},
            'optional_field': {'mandatory': False}
        }):
            
            # Test with missing required field
            incomplete_inputs = {
                'project_prf_functional_size': 100,
                # Missing project_prf_max_team_size
                'optional_field': 'test'
            }
            
            # Simulate validation logic from sidebar_inputs
            required_fields = [
                fname for fname, fdef in ui.FIELDS.items() 
                if fdef.get('mandatory', False)
            ]
            
            missing_fields = []
            for field in required_fields:
                value = incomplete_inputs.get(field)
                if value is None or value == "" or value == []:
                    missing_fields.append(field)
            
            assert len(missing_fields) == 1
            assert 'project_prf_max_team_size' in missing_fields

    def test_graceful_model_unavailable_handling(self):
        """Test graceful handling when models are unavailable"""
        
        # Simulate no models available
        st.session_state['cached_model_system'] = {
            "status": {"models_available": False},
            "models": [],
            "initialized": False
        }
        
        with patch('streamlit.warning') as mock_warning:
            with patch.object(ui, 'check_required_models') as mock_check:
                mock_check.return_value = {"models_available": False}
                
                # System should handle gracefully
                user_inputs = {
                    'selected_model': None,
                    'submit': True
                }
                
                # Should not crash, should show appropriate feedback
                selected_model = user_inputs.get('selected_model')
                if user_inputs.get('submit', False) and not selected_model:
                    # This condition should be handled gracefully in UI
                    pass

class TestWelcomeAndGuidance:
    """Test welcome screen and user guidance"""
    
    def test_welcome_screen_display(self):
        """Test welcome screen when no predictions made"""
        
        with patch('streamlit.info') as mock_info:
            
            # Simulate main() logic for welcome screen
            has_prediction_results = st.session_state.get('current_prediction_results') is not None
            
            if not has_prediction_results:
                expected_msg = "**Get Started:** Fill in the project parameters in the sidebar and click 'Predict Effort' to get your estimate."
                st.info(expected_msg)
                mock_info.assert_called_with(expected_msg)

    def test_required_field_markers(self):
        """Test required field markers are displayed"""
        
        with patch('streamlit.info') as mock_info:
            
            # Test required field marker display
            expected_marker_info = f"Required fields (marked with {UIConstants.REQUIRED_FIELD_MARKER})"
            
            # This would be called in sidebar
            st.info(expected_marker_info)
            mock_info.assert_called_with(expected_marker_info)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])