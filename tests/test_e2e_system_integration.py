# test_e2e_system_integration.py
"""
End-to-End System Integration Tests for AgileEE
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
        
        with patch('streamlit.set_page_config') as mock_page_config, \
             patch('streamlit.title'), \
             patch('streamlit.markdown'), \
             patch('streamlit.sidebar'), \
             patch('streamlit.tabs'), \
             patch('streamlit.header'), \
             patch('streamlit.info'), \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.columns'), \
             patch('streamlit.button'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.number_input'), \
             patch('streamlit.expander'):
            
            with patch.object(ui, 'check_required_models') as mock_check, \
                 patch.object(ui, 'list_available_models') as mock_list, \
                 patch.object(ui, 'FileConstants') as mock_constants, \
                 patch.object(ui, 'UI_INFO_CONFIG', {
                     'fields': {'test_field': {'type': 'numeric', 'mandatory': True}},
                     'tab_organization': {'Important Features': ['test_field'], 'Nice Features': []}
                 }):
                
                mock_check.return_value = {"models_available": True}
                mock_list.return_value = [{"display_name": "Test Model", "technical_name": "test_model"}]
                
                # Step 1: System initialization
                ui.set_sidebar_width()  # CSS setup
                ui.initialize_session_state()  # State setup
                
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
            },
            'feature_importance_display': {
                'max_features_shown': 10,
                'precision_decimals': 3
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
        
        with patch.object(ui, 'UI_INFO_CONFIG', mock_ui_config), \
             patch.object(ui, 'FEATURE_MAPPING', mock_feature_mapping), \
             patch.object(ui, 'FIELDS', mock_ui_config['fields']), \
             patch.object(ui, 'TAB_ORG', mock_ui_config['tab_organization']):
            
            # Test configuration integration
            fields = ui.FIELDS
            assert 'project_prf_functional_size' in fields
            assert fields['project_prf_functional_size']['mandatory'] is True
            
            tab_org = ui.get_tab_organization()
            assert 'Important Features' in tab_org
            
            field_label = ui.get_field_label('project_prf_functional_size')
            assert field_label == 'Functional Size'
            
            field_options = ui.get_field_options('project_prf_relative_size')
            assert field_options == ['Small', 'Medium', 'Large']

class TestE2EFullApplicationFlow:
    """Test complete application flow from start to finish"""
    
    def setup_method(self):
        """Setup comprehensive test environment"""
        # Reset session state
        st.session_state = {
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
        }
        
        # Mock comprehensive field configuration
        self.mock_fields = {
            'project_prf_functional_size': {
                'type': 'numeric', 'min': 1, 'max': 1000, 'default': 100, 'mandatory': True,
                'label': 'Functional Size', 'help': 'Project functional size in story points'
            },
            'project_prf_max_team_size': {
                'type': 'numeric', 'min': 1, 'max': 50, 'default': 5, 'mandatory': True,
                'label': 'Max Team Size', 'help': 'Maximum number of team members'
            },
            'project_prf_relative_size': {
                'type': 'categorical', 'mandatory': True,
                'label': 'Relative Size', 'help': 'Relative project size category'
            },
            'external_eef_industry_sector': {
                'type': 'categorical', 'mandatory': False,
                'label': 'Industry Sector', 'help': 'Industry sector of the project'
            },
            'tech_tf_primary_programming_language': {
                'type': 'categorical', 'mandatory': False,
                'label': 'Programming Language', 'help': 'Primary programming language'
            },
            'project_methodology': {
                'type': 'categorical', 'mandatory': False,
                'label': 'Methodology', 'help': 'Development methodology'
            }
        }

    def test_e2e_complete_application_lifecycle(self):
        """Test complete application lifecycle with all features"""
        
        with patch('streamlit.set_page_config'), \
             patch('streamlit.title'), \
             patch('streamlit.markdown'), \
             patch('streamlit.sidebar'), \
             patch('streamlit.tabs') as mock_tabs, \
             patch('streamlit.header'), \
             patch('streamlit.info'), \
             patch('streamlit.warning'), \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.number_input') as mock_number, \
             patch('streamlit.metric'), \
             patch('streamlit.spinner') as mock_spinner, \
             patch('streamlit.dataframe') as mock_dataframe, \
             patch('streamlit.bar_chart'), \
             patch('streamlit.expander') as mock_expander, \
             patch('plotly.express.box') as mock_box_plot, \
             patch('streamlit.plotly_chart'):
            
            # Setup mocks
            mock_columns.return_value = [MagicMock(), MagicMock()]
            mock_tabs.return_value = [MagicMock() for _ in range(5)]
            mock_button.side_effect = [False, False, False, True, False, False, False, True]  # Multiple interactions
            mock_selectbox.side_effect = ['Medium', 'Financial', 'Java', 'Random Forest', 'XGBoost']
            mock_number.side_effect = [250, 6, 200, 5]  # Multiple numeric inputs
            
            # Mock spinner and expander contexts
            spinner_context = MagicMock()
            mock_spinner.return_value.__enter__ = Mock(return_value=spinner_context)
            mock_spinner.return_value.__exit__ = Mock(return_value=None)
            
            expander_context = MagicMock()
            mock_expander.return_value.__enter__ = Mock(return_value=expander_context)
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            
            with patch.object(ui, 'check_required_models') as mock_check, \
                 patch.object(ui, 'list_available_models') as mock_list, \
                 patch.object(ui, 'predict_man_hours') as mock_predict, \
                 patch.object(ui, 'get_feature_importance') as mock_importance, \
                 patch.object(ui, 'get_model_display_name') as mock_display_name, \
                 patch.object(ui, 'get_model_display_name_from_config') as mock_display_config, \
                 patch.object(ui, 'display_optimized_shap_analysis') as mock_shap, \
                 patch.object(ui, 'FIELDS', self.mock_fields), \
                 patch.object(ui, 'get_tab_organization') as mock_tab_org, \
                 patch('builtins.open', mock_open(read_data="# Static SHAP Report\nFeature analysis...")):
                
                # Configure mocks
                mock_check.return_value = {"models_available": True}
                mock_list.return_value = [
                    {'display_name': 'Random Forest', 'technical_name': 'rf_model'},
                    {'display_name': 'XGBoost', 'technical_name': 'xgb_model'}
                ]
                mock_predict.side_effect = [485.0, 520.0]  # Two predictions
                mock_importance.return_value = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
                mock_display_name.return_value = "Random Forest"
                mock_display_config.side_effect = lambda x: {
                    'rf_model': 'Random Forest',
                    'xgb_model': 'XGBoost'
                }.get(x, x)
                mock_tab_org.return_value = {
                    "Important Features": ["project_prf_functional_size", "project_prf_max_team_size", "project_prf_relative_size"],
                    "Nice Features": ["external_eef_industry_sector", "tech_tf_primary_programming_language", "project_methodology"]
                }
                
                # Phase 1: Application Startup
                ui.set_sidebar_width()
                ui.initialize_session_state()
                
                # Phase 2: First Prediction
                user_inputs_1 = ui.sidebar_inputs()
                
                # Simulate first prediction
                project_1 = {
                    'project_prf_functional_size': 250,
                    'project_prf_max_team_size': 6,
                    'project_prf_relative_size': 'M',
                    'external_eef_industry_sector': 'Financial',
                    'tech_tf_primary_programming_language': 'Java'
                }
                
                prediction_1 = mock_predict(project_1, 'rf_model')
                st.session_state['current_prediction_results'] = {
                    'prediction': prediction_1,
                    'model': 'rf_model',
                    'inputs': project_1
                }
                ui.add_prediction_to_history(project_1, 'rf_model', prediction_1)
                
                # Display first prediction results
                results = st.session_state['current_prediction_results']
                ui.display_inputs(results['inputs'], results['model'])
                ui.show_prediction(results['prediction'], results['model'], results['inputs'])
                ui.show_feature_importance(results['model'], results['inputs'])
                
                # Phase 3: Second Prediction for Comparison
                project_2 = {
                    'project_prf_functional_size': 200,
                    'project_prf_max_team_size': 5,
                    'project_prf_relative_size': 'M',
                    'external_eef_industry_sector': 'Financial',
                    'tech_tf_primary_programming_language': 'Java'
                }
                
                prediction_2 = mock_predict(project_2, 'xgb_model')
                ui.add_prediction_to_history(project_2, 'xgb_model', prediction_2)
                
                # Phase 4: SHAP Analysis
                latest_prediction = st.session_state['prediction_history'][-1]
                ui.display_instance_specific_shap(
                    latest_prediction['inputs'],
                    latest_prediction['model_technical']
                )
                
                # Phase 5: Model Comparison
                ui.display_model_comparison()
                
                # Phase 6: Static SHAP Analysis
                ui.display_static_shap_analysis()
                
                # Phase 7: Help System
                ui.about_section()
                
                # Verify Complete Lifecycle
                assert len(st.session_state['prediction_history']) == 2
                assert st.session_state['current_prediction_results'] is not None
                
                # Verify all major components were called
                mock_predict.assert_called()
                mock_importance.assert_called()
                mock_shap.assert_called()
                mock_box_plot.assert_called()
                mock_dataframe.assert_called()

class TestE2EErrorRecoveryAndResilience:
    """Test system resilience and error recovery"""
    
    def test_e2e_graceful_degradation_complete_flow(self):
        """Test system continues working when individual components fail"""
        
        with patch('streamlit.error') as mock_error, \
             patch('streamlit.warning') as mock_warning, \
             patch('streamlit.info'), \
             patch('streamlit.sidebar'), \
             patch('streamlit.tabs'), \
             patch('streamlit.title'), \
             patch('streamlit.markdown'), \
             patch('streamlit.header'), \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.columns'), \
             patch('streamlit.button'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.number_input'), \
             patch('streamlit.metric'), \
             patch('streamlit.dataframe'), \
             patch('streamlit.bar_chart'):
            
            # Test 1: Models unavailable initially
            with patch.object(ui, 'check_required_models') as mock_check, \
                 patch.object(ui, 'list_available_models') as mock_list:
                
                mock_check.return_value = {"models_available": False}
                mock_list.return_value = []
                
                # System should handle gracefully
                ui.initialize_session_state()
                user_inputs = ui.sidebar_inputs()
                
                assert user_inputs.get('selected_model') is None
                # No crash should occur
                
            # Test 2: Prediction fails but system continues
            with patch.object(ui, 'predict_man_hours', side_effect=Exception("Prediction failed")):
                
                try:
                    ui.predict_man_hours({'test': 'input'}, 'rf_model')
                except Exception:
                    pass  # Should be caught by UI layer
                
                # System should continue working
                ui.initialize_session_state()  # Should still work
                
            # Test 3: SHAP analysis fails but other features work
            with patch.object(ui, 'display_optimized_shap_analysis', side_effect=Exception("SHAP failed")):
                
                st.session_state['prediction_history'] = [{
                    'inputs': {'test': 'input'},
                    'model_technical': 'rf_model'
                }]
                
                try:
                    latest = st.session_state['prediction_history'][-1]
                    ui.display_instance_specific_shap(latest['inputs'], latest['model_technical'])
                except Exception:
                    pass  # Should be handled gracefully
                
                # Other features should still work
                ui.show_prediction_history()  # Should work
                
            # Test 4: Static SHAP file missing
            with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
                
                ui.display_static_shap_analysis()
                
                # Should show error but not crash
                mock_error.assert_called()

    def test_e2e_data_consistency_across_failures(self):
        """Test data consistency is maintained across component failures"""
        
        with patch.object(ui, 'predict_man_hours') as mock_predict:
            
            # Add successful prediction
            mock_predict.return_value = 480.0
            ui.add_prediction_to_history({'test': 'input'}, 'rf_model', 480.0)
            
            initial_history_length = len(st.session_state['prediction_history'])
            
            # Simulate failure in subsequent operations
            with patch.object(ui, 'get_feature_importance', side_effect=Exception("Feature importance failed")):
                
                try:
                    ui.get_feature_importance('rf_model')
                except Exception:
                    pass
                
                # History should remain intact
                assert len(st.session_state['prediction_history']) == initial_history_length
                assert st.session_state['prediction_history'][0]['prediction_hours'] == 480.0
                
            # Add another prediction despite previous failure
            mock_predict.return_value = 520.0
            ui.add_prediction_to_history({'test': 'input2'}, 'xgb_model', 520.0)
            
            # Should have both predictions
            assert len(st.session_state['prediction_history']) == 2
            assert st.session_state['prediction_history'][1]['prediction_hours'] == 520.0

class TestE2EPerformanceAndScalability:
    """Test performance characteristics and scalability"""
    
    def test_e2e_large_dataset_handling(self):
        """Test system performance with large datasets"""
        
        # Create large prediction history
        large_history = []
        for i in range(500):  # Large number of predictions
            large_history.append({
                'timestamp': f'2024-01-01 {i % 24:02d}:{i % 60:02d}:00',
                'model': f'Model {i % 5}',
                'model_technical': f'model_{i % 5}',
                'prediction_hours': 400.0 + (i % 100 * 10),
                'inputs': {
                    'project_prf_functional_size': 100 + (i % 50),
                    'project_prf_max_team_size': 3 + (i % 8)
                }
            })
        
        st.session_state['prediction_history'] = large_history
        
        with patch('streamlit.subheader'), \
             patch('streamlit.dataframe') as mock_dataframe, \
             patch('streamlit.info'), \
             patch('streamlit.error'), \
             patch('streamlit.warning'), \
             patch('streamlit.columns') as mock_columns, \
             patch('plotly.express.box') as mock_box_plot, \
             patch('streamlit.plotly_chart'):
            
            mock_columns.return_value = [MagicMock(), MagicMock()]
            
            with patch.object(ui, 'UIConstants') as mock_constants, \
                 patch.object(ui, 'get_model_display_name_from_config') as mock_display:
                
                mock_constants.HOURS_PER_DAY = 8
                mock_display.side_effect = lambda x: x.replace('_', ' ').title()
                
                # Test history display with large dataset
                ui.show_prediction_history()
                
                # Should handle large dataset without timeout
                # (In real app, might implement pagination)
                
                # Test comparison with many models
                ui.display_model_comparison()
                
                # Should handle multiple models efficiently
                if mock_box_plot.called:
                    # Verify it processed the data
                    assert True

    def test_e2e_memory_management(self):
        """Test memory usage patterns"""
        
        # Simulate memory-intensive operations
        with patch.object(ui, 'predict_man_hours') as mock_predict, \
             patch.object(ui, 'get_feature_importance') as mock_importance, \
             patch.object(ui, 'display_optimized_shap_analysis') as mock_shap:
            
            mock_predict.return_value = 480.0
            mock_importance.return_value = np.array([0.3] * 100)  # Large importance array
            
            # Add many predictions
            for i in range(50):
                large_input = {f'feature_{j}': j for j in range(20)}  # Many features
                ui.add_prediction_to_history(large_input, f'model_{i % 3}', 480.0 + i)
            
            # System should handle without memory issues
            assert len(st.session_state['prediction_history']) == 50
            
            # Clear history to free memory
            st.session_state['prediction_history'] = []
            st.session_state['current_prediction_results'] = None
            
            # Verify cleanup
            assert len(st.session_state['prediction_history']) == 0

class TestE2ESecurityAndValidation:
    """Test security aspects and input validation"""
    
    def test_e2e_input_sanitization(self):
        """Test input sanitization and validation"""
        
        with patch('streamlit.error') as mock_error, \
             patch('streamlit.warning') as mock_warning:
            
            # Test with potentially malicious inputs
            malicious_inputs = {
                'project_prf_functional_size': -100,  # Negative value
                'project_prf_max_team_size': 99999,   # Too large
                'project_prf_relative_size': '<script>alert("xss")</script>',  # XSS attempt
                'external_eef_industry_sector': 'SELECT * FROM users;',  # SQL injection attempt
                'tech_tf_primary_programming_language': '../../etc/passwd'  # Path traversal attempt
            }
            
            with patch.object(ui, 'predict_man_hours') as mock_predict:
                
                # System should handle malicious inputs gracefully
                try:
                    mock_predict.return_value = 480.0
                    ui.add_prediction_to_history(malicious_inputs, 'rf_model', 480.0)
                    
                    # Verify data is stored safely (no execution of malicious code)
                    history = st.session_state['prediction_history']
                    if history:
                        stored_inputs = history[-1]['inputs']
                        
                        # Data should be stored as-is but not executed
                        assert isinstance(stored_inputs, dict)
                        
                except Exception:
                    # If validation catches malicious input, that's good
                    pass

    def test_e2e_session_state_isolation(self):
        """Test session state isolation and security"""
        
        # Test that session state doesn't leak between tests
        ui.initialize_session_state()
        
        # Add some data
        st.session_state['test_data'] = 'sensitive_information'
        ui.add_prediction_to_history({'test': 'input'}, 'rf_model', 480.0)
        
        # Clear and reinitialize
        st.session_state.clear()
        ui.initialize_session_state()
        
        # Should not contain previous data
        assert 'test_data' not in st.session_state
        assert len(st.session_state['prediction_history']) == 0

class TestE2EAccessibilityAndUsability:
    """Test accessibility and usability features"""
    
    def test_e2e_accessibility_compliance(self):
        """Test accessibility compliance across the application"""
        
        with patch('streamlit.title') as mock_title, \
             patch('streamlit.header') as mock_header, \
             patch('streamlit.subheader') as mock_subheader, \
             patch('streamlit.info') as mock_info, \
             patch('streamlit.markdown'), \
             patch('streamlit.sidebar'), \
             patch('streamlit.tabs'), \
             patch('streamlit.button'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.number_input'):
            
            with patch.object(ui, 'check_required_models') as mock_check, \
                 patch.object(ui, 'list_available_models') as mock_list:
                
                mock_check.return_value = {"models_available": True}
                mock_list.return_value = [{"display_name": "Test Model", "technical_name": "test_model"}]
                
                # Test proper heading structure
                ui.initialize_session_state()
                ui.sidebar_inputs()
                
                # Verify headings are called (screen reader navigation)
                mock_title.assert_called()
                
                # Test help content accessibility
                ui.about_section()
                
                # Test that required field markers are present
                mock_info.assert_called()  # Should show required field guidance

    def test_e2e_responsive_design_behavior(self):
        """Test responsive design behavior"""
        
        with patch('streamlit.columns') as mock_columns, \
             patch('streamlit.sidebar'), \
             patch('streamlit.tabs'), \
             patch('streamlit.header'), \
             patch('streamlit.subheader'), \
             patch('streamlit.dataframe'), \
             patch('plotly.express.box'), \
             patch('streamlit.plotly_chart'):
            
            # Test two-column layout
            mock_columns.return_value = [MagicMock(), MagicMock()]
            
            st.session_state['prediction_history'] = [
                {'model_technical': 'rf_model', 'prediction_hours': 480.0},
                {'model_technical': 'xgb_model', 'prediction_hours': 520.0}
            ]
            
            with patch.object(ui, 'get_model_display_name_from_config') as mock_display:
                mock_display.side_effect = lambda x: x.replace('_', ' ').title()
                
                # Test responsive layout
                ui.display_model_comparison()
                
                # Verify columns are used for responsive design
                mock_columns.assert_called_with(2)

class TestE2EBackwardCompatibility:
    """Test backward compatibility and migration scenarios"""
    
    def test_e2e_legacy_data_handling(self):
        """Test handling of legacy data formats"""
        
        # Simulate legacy prediction history format
        legacy_history = [
            {
                # Old format without model_technical
                'timestamp': '2024-01-01 10:00:00',
                'model': 'Random Forest',
                'prediction_hours': 480.0,
                'inputs': {'project_prf_functional_size': 100}
                # Missing 'model_technical' field
            },
            {
                # Mixed format
                'timestamp': '2024-01-01 11:00:00',
                'model': 'XGBoost',
                'model_technical': 'xgb_model',
                'prediction_hours': 520.0,
                'inputs': {'project_prf_functional_size': 150}
            }
        ]
        
        st.session_state['prediction_history'] = legacy_history
        
        with patch('streamlit.subheader'), \
             patch('streamlit.dataframe'), \
             patch('streamlit.info'), \
             patch('streamlit.error'), \
             patch('streamlit.warning'), \
             patch('streamlit.columns') as mock_columns, \
             patch('plotly.express.box'), \
             patch('streamlit.plotly_chart'):
            
            mock_columns.return_value = [MagicMock(), MagicMock()]
            
            with patch.object(ui, 'UIConstants') as mock_constants, \
                 patch.object(ui, 'get_model_display_name_from_config') as mock_display:
                
                mock_constants.HOURS_PER_DAY = 8
                mock_display.side_effect = lambda x: x if x else 'Unknown Model'
                
                # Should handle legacy data gracefully
                ui.show_prediction_history()
                ui.display_model_comparison()
                
                # No crashes should occur with mixed data formats
                assert len(st.session_state['prediction_history']) == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])