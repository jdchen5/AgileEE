# tests/conftest.py
"""
Shared pytest configuration and fixtures for UI simplification tests.
This file provides common test setup, mock data, and utilities.
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
from contextlib import ExitStack
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the project root to the path so tests can import the main modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the modules under test
from agileee import ui
from agileee.constants import UIConstants, FileConstants

# ==================== PYTEST CONFIGURATION ====================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "tab: marks tests for specific tabs")

# ==================== COMMON FIXTURES ====================

@pytest.fixture
def clean_session_state():
    """Provide a clean Streamlit session state for each test."""
    # Save original state
    original_state = dict(st.session_state) if hasattr(st, 'session_state') else {}
    
    # Reset to clean state
    st.session_state.clear()
    st.session_state.update({
        'prediction_history': [],
        'comparison_results': [],
        'form_attempted': False,
        'prf_size_label2code': {},
        'prf_size_code2mid': {},
        'current_shap_values': None,
        'current_model_explainer': None,
        'last_prediction_inputs': None
    })
    
    yield st.session_state
    
    # Restore original state
    st.session_state.clear()
    st.session_state.update(original_state)

@pytest.fixture
def mock_streamlit_components():
    """Mock all Streamlit UI components to prevent actual rendering during tests."""
    with ExitStack() as stack:
        # Define all patches to apply
        patches = {
            'title': stack.enter_context(patch('streamlit.title')),
            'header': stack.enter_context(patch('streamlit.header')),
            'subheader': stack.enter_context(patch('streamlit.subheader')),
            'markdown': stack.enter_context(patch('streamlit.markdown')),
            'info': stack.enter_context(patch('streamlit.info')),
            'warning': stack.enter_context(patch('streamlit.warning')),
            'error': stack.enter_context(patch('streamlit.error')),
            'success': stack.enter_context(patch('streamlit.success')),
            'tabs': stack.enter_context(patch('streamlit.tabs')),
            'columns': stack.enter_context(patch('streamlit.columns')),
            'expander': stack.enter_context(patch('streamlit.expander')),
            'sidebar': stack.enter_context(patch('streamlit.sidebar')),
            'button': stack.enter_context(patch('streamlit.button')),
            'selectbox': stack.enter_context(patch('streamlit.selectbox')),
            'number_input': stack.enter_context(patch('streamlit.number_input')),
            'text_input': stack.enter_context(patch('streamlit.text_input')),
            'checkbox': stack.enter_context(patch('streamlit.checkbox')),
            'divider': stack.enter_context(patch('streamlit.divider')),
            'metric': stack.enter_context(patch('streamlit.metric')),
            'dataframe': stack.enter_context(patch('streamlit.dataframe')),
            'bar_chart': stack.enter_context(patch('streamlit.bar_chart')),
            'plotly_chart': stack.enter_context(patch('streamlit.plotly_chart')),
            'spinner': stack.enter_context(patch('streamlit.spinner')),
            'file_uploader': stack.enter_context(patch('streamlit.file_uploader')),
            'download_button': stack.enter_context(patch('streamlit.download_button'))
        }
        
        # Configure default behaviors
        patches['tabs'].return_value = [MagicMock() for _ in range(5)]
        patches['columns'].return_value = [MagicMock(), MagicMock()]
        patches['button'].return_value = False
        patches['selectbox'].return_value = "default_value"
        patches['number_input'].return_value = 100
        patches['text_input'].return_value = "test_value"
        patches['checkbox'].return_value = True
        
        # Configure context managers
        for context_name in ['expander', 'sidebar', 'spinner']:
            patches[context_name].return_value.__enter__ = Mock(return_value=MagicMock())
            patches[context_name].return_value.__exit__ = Mock(return_value=None)
        
        yield patches

@pytest.fixture
def sample_user_inputs():
    """Provide sample user inputs for testing."""
    return {
        'project_prf_year_of_project': 2024,
        'project_prf_functional_size': 100,
        'project_prf_max_team_size': 5,
        'external_eef_industry_sector': 'Financial',
        'tech_tf_primary_programming_language': 'Java',
        'project_prf_relative_size': 'M',
        'project_prf_development_type': 'New Development',
        'tech_tf_language_type': 'Object Oriented',
        'project_prf_application_type': 'Web Application',
        'external_eef_organisation_type': 'Banking',
        'tech_tf_architecture': 'Client Server',
        'tech_tf_development_platform': 'Web',
        'project_prf_team_size_group': '3-10',
        'tech_tf_server_roles': 'Application Server',
        'tech_tf_client_roles': 'Web Browser',
        'tech_tf_web_development': True,
        'tech_tf_dbms_used': True,
        'process_pmf_prototyping_used': False,
        'project_prf_case_tool_used': True,
        'process_pmf_docs': 5,
        'people_prf_project_user_involvement': 3
    }

@pytest.fixture
def sample_prediction_history():
    """Provide sample prediction history for testing."""
    return [
        {
            'timestamp': '2024-01-01 10:00:00',
            'model': 'Random Forest',
            'model_technical': 'rf_model',
            'prediction_hours': 480.0,
            'inputs': {
                'project_prf_functional_size': 100,
                'project_prf_max_team_size': 5,
                'external_eef_industry_sector': 'Financial'
            }
        },
        {
            'timestamp': '2024-01-01 11:00:00',
            'model': 'XGBoost',
            'model_technical': 'xgb_model',
            'prediction_hours': 520.0,
            'inputs': {
                'project_prf_functional_size': 150,
                'project_prf_max_team_size': 7,
                'external_eef_industry_sector': 'Banking'
            }
        }
    ]

@pytest.fixture
def mock_available_models():
    """Provide mock available models for testing."""
    return [
        {
            'technical_name': 'rf_model',
            'display_name': 'Random Forest'
        },
        {
            'technical_name': 'xgb_model', 
            'display_name': 'XGBoost'
        },
        {
            'technical_name': 'lr_model',
            'display_name': 'Linear Regression'
        }
    ]

@pytest.fixture
def mock_ui_fields():
    """Provide mock UI fields configuration for testing."""
    return {
        'project_prf_functional_size': {
            'type': 'numeric',
            'label': 'Functional Size',
            'min': 1,
            'max': 10000,
            'default': 100,
            'mandatory': True,
            'help': 'Size of the project in function points'
        },
        'project_prf_max_team_size': {
            'type': 'numeric', 
            'label': 'Max Team Size',
            'min': 1,
            'max': 50,
            'default': 5,
            'mandatory': True,
            'help': 'Maximum number of people on the team'
        },
        'external_eef_industry_sector': {
            'type': 'categorical',
            'label': 'Industry Sector',
            'options': ['Financial', 'Banking', 'Insurance', 'Healthcare'],
            'default': 'Financial',
            'mandatory': False,
            'help': 'The industry sector of the organization'
        },
        'tech_tf_web_development': {
            'type': 'boolean',
            'label': 'Web Development',
            'default': False,
            'mandatory': False,
            'help': 'Is this a web development project?'
        }
    }

# ==================== MOCK PATCHES ====================

@pytest.fixture
def mock_models_module():
    """Mock the models module functions."""
    with ExitStack() as stack:
        patches = {
            'models_available': stack.enter_context(patch.object(ui, 'MODELS_AVAILABLE', True)),
            'predict_man_hours': stack.enter_context(patch.object(ui, 'predict_man_hours')),
            'list_available_models': stack.enter_context(patch.object(ui, 'list_available_models')),
            'check_required_models': stack.enter_context(patch.object(ui, 'check_required_models')),
            'get_feature_importance': stack.enter_context(patch.object(ui, 'get_feature_importance')),
            'get_model_display_name': stack.enter_context(patch.object(ui, 'get_model_display_name')),
            'get_trained_model': stack.enter_context(patch.object(ui, 'get_trained_model'))
        }
        
        # Configure default behaviors
        patches['predict_man_hours'].return_value = 480.0
        patches['list_available_models'].return_value = [
            {'technical_name': 'rf_model', 'display_name': 'Random Forest'},
            {'technical_name': 'xgb_model', 'display_name': 'XGBoost'}
        ]
        patches['check_required_models'].return_value = {'models_available': True}
        patches['get_feature_importance'].return_value = np.array([0.3, 0.2, 0.15, 0.1, 0.05])
        patches['get_model_display_name'].side_effect = lambda x: x.replace('_', ' ').title()
        patches['get_trained_model'].return_value = MagicMock()
        
        yield patches

@pytest.fixture
def mock_shap_module():
    """Mock the SHAP analysis module."""
    with ExitStack() as stack:
        patches = {
            'display_instance_specific_shap': stack.enter_context(patch.object(ui, 'display_instance_specific_shap')),
            'display_static_shap_analysis': stack.enter_context(patch.object(ui, 'display_static_shap_analysis'))
        }
        
        patches['display_instance_specific_shap'].return_value = None
        patches['display_static_shap_analysis'].return_value = None
        
        yield patches

# ==================== UTILITY FUNCTIONS ====================

def assert_no_config_references(function_or_source):
    """Utility function to assert no configuration management references exist."""
    if callable(function_or_source):
        import inspect
        try:
            source = inspect.getsource(function_or_source)
        except (OSError, TypeError):
            return  # Can't get source, skip check
    else:
        source = str(function_or_source)
    
    forbidden_patterns = [
        'save_config',
        'load_config', 
        'config_name',
        'uploaded_file',
        'file_uploader',
        'download_button',
        'make_current_config_json',
        'make_history_json',
        'load_configuration_from_data'
    ]
    
    for pattern in forbidden_patterns:
        assert pattern not in source, f"Found forbidden config reference: {pattern}"

def assert_streamlit_component_called(mock_components, component_name, expected_calls=None):
    """Utility function to assert Streamlit components were called correctly."""
    component = mock_components.get(component_name)
    assert component is not None, f"Component {component_name} not found in mocks"
    
    if expected_calls is not None:
        assert component.call_count >= expected_calls, \
            f"Expected {component_name} to be called at least {expected_calls} times, got {component.call_count}"
    else:
        assert component.called, f"Expected {component_name} to be called"

def create_mock_file_content(content_type="markdown"):
    """Create mock file content for testing file loading."""
    if content_type == "markdown":
        return """
# Test SHAP Analysis

## Feature Importance
- Feature 1: High importance
- Feature 2: Medium importance

## Model Performance
Good performance across all models.
"""
    elif content_type == "html":
        return """
<h1>Test SHAP Analysis</h1>
<p>HTML content for testing</p>
"""
    else:
        return "Plain text content for testing"

# ==================== PARAMETRIZED TEST DATA ====================

@pytest.fixture(params=[
    "rf_model",
    "xgb_model", 
    "lr_model",
    "svm_model"
])
def model_name(request):
    """Parametrized fixture for different model names."""
    return request.param

@pytest.fixture(params=[
    {"project_prf_functional_size": 50, "expected_low": True},
    {"project_prf_functional_size": 100, "expected_low": False}, 
    {"project_prf_functional_size": 500, "expected_low": False}
])
def project_size_scenarios(request):
    """Parametrized fixture for different project size scenarios."""
    return request.param