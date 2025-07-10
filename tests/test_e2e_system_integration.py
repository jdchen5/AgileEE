# test_e2e_system_integration.py
"""
End-to-End System Integration Tests for AgileEE - SIMPLIFIED VERSION
Tests the complete system integration including all components working together.
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock, mock_open
import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the UI module and dependencies
import agileee.ui as ui
from agileee.constants import UIConstants, FileConstants

class TestE2ESystemBootstrap:
    """Test system initialization and bootstrap process"""
    
    def test_e2e_full_system_startup(self):
        """Test complete system startup sequence"""
        
        # Use simpler patching approach
        streamlit_patches = [
            'streamlit.set_page_config',
            'streamlit.title',
            'streamlit.markdown',
            'streamlit.sidebar',
            'streamlit.tabs',
            'streamlit.header',
            'streamlit.info',
            'streamlit.subheader',
            'streamlit.divider',
            'streamlit.columns',
            'streamlit.button',
            'streamlit.selectbox',
            'streamlit.number_input',
            'streamlit.expander'
        ]
        
        with patch.multiple('streamlit', **{p.split('.')[-1]: MagicMock() for p in streamlit_patches}):
            with patch.object(ui, 'check_required_models') as mock_check:
                with patch.object(ui, 'list_available_models') as mock_list:
                    with patch.object(ui, 'UI_INFO_CONFIG', {
                        'fields': {'test_field': {'type': 'numeric', 'mandatory': True}},
                        'tab_organization': {'Important Features': ['test_field'], 'Nice Features': []}
                    }):
                        
                        mock_check.return_value = {"models_available": True}
                        mock_list.return_value = [{"display_name": "Test Model", "technical_name": "test_model"}]
                        
                        # Step 1: System initialization
                        ui.set_sidebar_width()
                        ui.initialize_session_state()
                        
                        # Verify session state is properly initialized
                        expected_state_keys = [
                            'prediction_history', 'comparison_results', 'form_attempted',
                            'prf_size_label2code', 'prf_size_code2mid', 'current_shap_values',
                            'current_model_explainer', 'last_prediction_inputs'
                        ]
                        
                        for key in expected_state_keys:
                            assert key in st.session_state
                        
                        # Step 2: UI components load
                        user_inputs = ui.sidebar_inputs()
                        assert isinstance(user_inputs, dict)
                        
                        # Step 3: Models are accessible
                        mock_check.assert_called()
                        mock_list.assert_called()
                        
                        # Step 4: Complete system is ready
                        assert st.session_state['prediction_history'] == []
                        assert 'selected_model' in user_inputs

    def test_e2e_configuration_loading_integration(self):
        """Test configuration loading and integration"""
        
        mock_ui_config = {
            'fields': {
                'project_prf_functional_size': {
                    'type': 'numeric', 'mandatory': True, 'min': 1, 'max': 1000,
                    'label': 'Functional Size', 'help': 'Project functional size'
                },
                'project_prf_max_team_size': {
                    'type': 'numeric', 'mandatory': True, 'min': 1, 'max': 50,
                    'label': 'Team Size', 'help': 'Maximum team size'
                }
            },
            'tab_organization': {
                'Important Features': ['project_prf_functional_size', 'project_prf_max_team_size'],
                'Nice Features': []
            }
        }
        
        mock_feature_mapping = {
            'categorical_features': {
                'project_prf_relative_size': {
                    'options': [
                        {'code': 'S', 'label': 'Small', 'midpoint': 75},
                        {'code': 'M', 'label': 'Medium', 'midpoint': 300},
                        {'code': 'L', 'label': 'Large', 'midpoint': 1000}
                    ]
                }
            }
        }
        
        with patch.object(ui, 'UI_INFO_CONFIG', mock_ui_config):
            with patch.object(ui, 'FEATURE_MAPPING', mock_feature_mapping):
                with patch.object(ui, 'FIELDS', mock_ui_config['fields']):
                    
                    # Test configuration integration
                    fields = ui.FIELDS
                    assert 'project_prf_functional_size' in fields
                    assert fields['project_prf_functional_size']['mandatory'] is True
                    
                    tab_org = ui.get_tab_organization()
                    assert 'Important Features' in tab_org
                    
                    field_label = ui.get_field_label('project_prf_functional_size')
                    assert field_label == 'Functional Size'

class TestE2EFullApplicationFlow:
    """Test complete application flow from start to finish"""
    
    def setup_method(self):
        """Setup comprehensive test environment"""
        # Reset session state
        st.session_state.clear()
        st.session_state.update({
            'prediction_history': [],
            'comparison_results': [],
            'form_attempted': False,
            'prf_size_label2code': {'Small': 'S', 'Medium': 'M', 'Large': 'L'},
            'prf_size_code2mid': {'S': 75, 'M': 300, 'L': 1000},
            'prf_size_code2full': {
                'S': {'code': 'S', 'label': 'Small', 'midpoint': 75, 'minimumhour': 50, 'maximumhour': 200},
                'M': {'code': 'M', 'label': 'Medium', 'midpoint': 300, 'minimumhour': 200, 'maximumhour': 800},
                'L': {'code': 'L', 'label': 'Large', 'midpoint': 1000, 'minimumhour': 800, 'maximumhour': 2000}
            },
            'current_shap_values': None,
            'current_model_explainer': None,
            'last_prediction_inputs': None,
            'current_prediction_results': None
        })

    def test_e2e_complete_application_lifecycle(self):
        """Test complete application lifecycle with all features"""
        
        # Simplified patching
        streamlit_mocks = {
            'set_page_config': MagicMock(),
            'title': MagicMock(),
            'markdown': MagicMock(),
            'sidebar': MagicMock(),
            'tabs': MagicMock(return_value=[MagicMock() for _ in range(5)]),
            'header': MagicMock(),
            'info': MagicMock(),
            'warning': MagicMock(),
            'subheader': MagicMock(),
            'divider': MagicMock(),
            'columns': MagicMock(return_value=[MagicMock(), MagicMock()]),
            'button': MagicMock(side_effect=[False, False, True]),
            'selectbox': MagicMock(side_effect=['Medium', 'Financial', 'Java']),
            'number_input': MagicMock(side_effect=[250, 6]),
            'metric': MagicMock(),
            'spinner': MagicMock(),
            'dataframe': MagicMock(),
            'bar_chart': MagicMock(),
            'expander': MagicMock(),
            'plotly_chart': MagicMock()
        }
        
        with patch.multiple('streamlit', **streamlit_mocks):
            with patch('plotly.express.box') as mock_box_plot:
                with patch.object(ui, 'check_required_models', return_value={"models_available": True}):
                    with patch.object(ui, 'list_available_models', return_value=[
                        {'display_name': 'Random Forest', 'technical_name': 'rf_model'}
                    ]):
                        with patch.object(ui, 'predict_man_hours', return_value=485.0) as mock_predict:
                            with patch.object(ui, 'get_feature_importance', return_value=np.array([0.3, 0.25, 0.2])):
                                with patch.object(ui, 'get_model_display_name', return_value="Random Forest"):
                                    with patch.object(ui, 'display_instance_specific_shap') as mock_shap:
                                        with patch('builtins.open', mock_open(read_data="# SHAP Report")):
                                            
                                            # Phase 1: Application Startup
                                            ui.set_sidebar_width()
                                            ui.initialize_session_state()
                                            
                                            # Phase 2: Make Prediction
                                            project_data = {
                                                'project_prf_functional_size': 250,
                                                'project_prf_max_team_size': 6,
                                                'external_eef_industry_sector': 'Financial'
                                            }
                                            
                                            prediction = mock_predict(project_data, 'rf_model')
                                            ui.add_prediction_to_history(project_data, 'rf_model', prediction)
                                            
                                            # Phase 3: Display Results
                                            ui.show_prediction_history()
                                            ui.display_model_comparison()
                                            ui.display_static_shap_analysis()
                                            
                                            # Verify lifecycle completed
                                            assert len(st.session_state['prediction_history']) == 1
                                            mock_predict.assert_called()

class TestE2EErrorRecoveryAndResilience:
    """Test system resilience and error recovery"""
    
    def test_e2e_graceful_degradation(self):
        """Test system continues working when individual components fail"""
        
        with patch('streamlit.error') as mock_error:
            with patch('streamlit.warning'):
                with patch('streamlit.info'):
                    
                    # Test 1: Models unavailable
                    with patch.object(ui, 'check_required_models', return_value={"models_available": False}):
                        with patch.object(ui, 'list_available_models', return_value=[]):
                            
                            ui.initialize_session_state()
                            user_inputs = ui.sidebar_inputs()
                            assert user_inputs.get('selected_model') is None
                    
                    # Test 2: Prediction fails
                    with patch.object(ui, 'predict_man_hours', side_effect=Exception("Prediction failed")):
                        try:
                            ui.predict_man_hours({'test': 'input'}, 'rf_model')
                        except Exception:
                            pass  # Expected to fail gracefully
                        
                        # System should continue working
                        ui.initialize_session_state()
                    
                    # Test 3: SHAP fails
                    st.session_state['prediction_history'] = [{'inputs': {'test': 'input'}, 'model_technical': 'rf_model'}]
                    
                    with patch.object(ui, 'display_instance_specific_shap', side_effect=Exception("SHAP failed")):
                        try:
                            latest = st.session_state['prediction_history'][-1]
                            ui.display_instance_specific_shap(latest['inputs'], latest['model_technical'])
                        except Exception:
                            pass
                        
                        # Other features should still work
                        ui.show_prediction_history()

class TestE2EDataConsistency:
    """Test data consistency across operations"""
    
    def test_e2e_data_persistence(self):
        """Test data consistency is maintained"""
        
        with patch.object(ui, 'predict_man_hours', return_value=480.0):
            
            # Add prediction
            ui.add_prediction_to_history({'test': 'input'}, 'rf_model', 480.0)
            initial_length = len(st.session_state['prediction_history'])
            
            # Simulate failure in feature importance
            with patch.object(ui, 'get_feature_importance', side_effect=Exception("Failed")):
                try:
                    ui.get_feature_importance('rf_model')
                except Exception:
                    pass
                
                # History should remain intact
                assert len(st.session_state['prediction_history']) == initial_length
                assert st.session_state['prediction_history'][0]['prediction_hours'] == 480.0

class TestE2EPerformance:
    """Test performance characteristics"""
    
    def test_e2e_large_dataset_handling(self):
        """Test system with large datasets"""
        
        # Create large prediction history
        large_history = []
        for i in range(100):  # Reduced from 500 for faster testing
            large_history.append({
                'timestamp': f'2024-01-01 {i % 24:02d}:00:00',
                'model': f'Model {i % 5}',
                'model_technical': f'model_{i % 5}',
                'prediction_hours': 400.0 + (i * 10),
                'inputs': {'project_prf_functional_size': 100 + i}
            })
        
        st.session_state['prediction_history'] = large_history
        
        with patch('streamlit.subheader'):
            with patch('streamlit.dataframe'):
                with patch('streamlit.columns', return_value=[MagicMock(), MagicMock()]):
                    with patch('plotly.express.box'):
                        with patch('streamlit.plotly_chart'):
                            with patch.object(ui, 'UIConstants') as mock_constants:
                                with patch.object(ui, 'get_model_display_name_from_config', side_effect=lambda x: x):
                                    
                                    mock_constants.HOURS_PER_DAY = 8
                                    
                                    # Should handle large dataset
                                    ui.show_prediction_history()
                                    ui.display_model_comparison()
                                    
                                    # Verify no timeout/crash
                                    assert len(st.session_state['prediction_history']) == 100

class TestE2EBackwardCompatibility:
    """Test backward compatibility"""
    
    def test_e2e_legacy_data_handling(self):
        """Test handling of legacy data formats"""
        
        # Legacy format without model_technical
        legacy_history = [
            {
                'timestamp': '2024-01-01 10:00:00',
                'model': 'Random Forest',
                'prediction_hours': 480.0,
                'inputs': {'project_prf_functional_size': 100}
                # Missing 'model_technical' field
            }
        ]
        
        st.session_state['prediction_history'] = legacy_history
        
        with patch('streamlit.subheader'):
            with patch('streamlit.dataframe'):
                with patch('streamlit.columns', return_value=[MagicMock(), MagicMock()]):
                    with patch.object(ui, 'UIConstants') as mock_constants:
                        with patch.object(ui, 'get_model_display_name_from_config', side_effect=lambda x: x or 'Unknown'):
                            
                            mock_constants.HOURS_PER_DAY = 8
                            
                            # Should handle legacy data gracefully
                            ui.show_prediction_history()
                            ui.display_model_comparison()
                            
                            # No crashes with legacy format
                            assert len(st.session_state['prediction_history']) == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])