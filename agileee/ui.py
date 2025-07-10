# ui.py - Simplified version without configuration management
"""
Streamlit UI for ML Project Effort Estimator with simplified interface
This module provides a user interface for estimating project effort using machine learning models.
It includes form inputs, model selection, prediction, and SHAP analysis.
"""

import streamlit as st
from functools import lru_cache
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import yaml
from datetime import datetime
import shap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import traceback
from agileee.constants import FileConstants, UIConstants, PipelineConstants
from agileee.config_loader import ConfigLoader

from agileee.shap_analysis import (
    get_shap_analysis_results,
    get_shap_explainer_optimized,
    clear_explainer_cache,
    get_cache_info   
)

# -------------- MODEL IMPORTS AND FALLBACKS ----------------

try:
    from agileee.models import (
        predict_man_hours,
        list_available_models,
        check_required_models,
        get_feature_importance,
        get_model_display_name,
        get_model_display_name_from_config,
        get_trained_model,
        prepare_input_data,
        prepare_features_for_model,
        load_model_display_names,
        load_preprocessing_pipeline
    )
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import from models.py: {e}")
    MODELS_AVAILABLE = False

    # Define stub functions so UI doesn't crash
    def predict_with_training_features_optimized(inputs, model):
        return None
    def predict_man_hours_direct(inputs, model):
        return None
    def list_available_models():
        return []
    def check_required_models():
        return {"models_available": False}
    def get_trained_model(model_name):
        return None
    def prepare_input_data(inputs):
        return None

DEBUG_MODE = os.getenv('STREAMLIT_DEBUG', 'true').lower() == 'true'



@st.cache_data
def load_ui_config_cached():
    """Cache UI configuration loading - same logic as before but cached"""
    ui_config_path = os.path.join(FileConstants.CONFIG_FOLDER, FileConstants.UI_INFO_FILE)
    UI_INFO_CONFIG = ConfigLoader.load_yaml_config(ui_config_path)
    if UI_INFO_CONFIG is None:
        UI_INFO_CONFIG = {}
        print(f"Warning: Could not load UI configuration from {ui_config_path}. Using empty config.")
    print("DEBUG: UI_INFO_CONFIG loaded successfully")
    return UI_INFO_CONFIG

@st.cache_data  
def load_feature_mapping_cached():
    """Cache feature mapping loading - same logic as before but cached"""
    feature_mapping_path = os.path.join(FileConstants.CONFIG_FOLDER, FileConstants.FEATURE_MAPPING_FILE)
    FEATURE_MAPPING = ConfigLoader.load_yaml_config(feature_mapping_path)
    if FEATURE_MAPPING is None:
        FEATURE_MAPPING = {}
        print(f"Warning: Could not load feature mapping from {feature_mapping_path}. Using empty mapping.")
    return FEATURE_MAPPING

@st.cache_resource
def initialize_model_system_cached():
    """Cache model system - fixes the repetitive loading issue"""
    try:
        print("DEBUG: Looking for models in: models")  # This will only print ONCE now
        print("DEBUG: Folder exists: True")
        # Your existing model loading logic here
        model_status = check_required_models()
        available_models = list_available_models() if model_status.get("models_available", False) else []
        print(f"DEBUG: Final model list: {len(available_models)} models found")
        return {
            "status": model_status,
            "models": available_models,
            "initialized": True
        }
    except Exception as e:
        return {"status": {"models_available": False}, "models": [], "initialized": False, "error": str(e)}

def debug_print(message):
    if DEBUG_MODE:
        print(f"DEBUG: {message}")

print("DEBUG: ui.py execution started")
print("DEBUG: About to load configurations...")

# --------------------- CONFIG LOADING ---------------------
# OPTIMIZED CONFIG LOADING - Load once and cache

# Initialize configs

UI_INFO_CONFIG = load_ui_config_cached()
FIELDS = UI_INFO_CONFIG.get('fields', {})
print("DEBUG: FIELDS loaded successfully")

TAB_ORG = UI_INFO_CONFIG.get('tab_organization', {})
print("DEBUG: TAB_ORG loaded successfully")

# Keep all your existing assignments exactly as they are:
UI_BEHAVIOR = UI_INFO_CONFIG.get('ui_behavior', {})
FEATURE_IMPORTANCE_DISPLAY = UI_INFO_CONFIG.get('feature_importance_display', {})
PREDICTION_THRESHOLDS = UI_INFO_CONFIG.get('prediction_thresholds', {})
DISPLAY_CONFIG = UI_INFO_CONFIG.get('display_config', {})


FEATURE_MAPPING = load_feature_mapping_cached()

CATEGORICAL_MAPPING = FEATURE_MAPPING.get('categorical_features', {})

IMPORTANT_TABS = "Important Features"
NICE_TABS = "Nice Features"

print("DEBUG: About to define functions...")

def set_sidebar_width():
    """Minimal CSS for sidebar width only"""
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        width: 350px !important;
        min-width: 350px !important;
        max-width: 350px !important;
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_configs_once():
    """Initialize configs once using session state"""
    if 'configs_loaded' not in st.session_state:
        st.session_state.UI_INFO_CONFIG = load_ui_config_cached() or {}
        st.session_state.FEATURE_MAPPING = load_feature_mapping_cached() or {}
        st.session_state.FIELDS = st.session_state.UI_INFO_CONFIG.get('fields', {})
        st.session_state.TAB_ORG = st.session_state.UI_INFO_CONFIG.get('tab_organization', {})
        st.session_state.configs_loaded = True
        print("DEBUG: Configs loaded and cached in session state")


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    defaults = {
        'prediction_history': [],
        'comparison_results': [],
        'form_attempted': False,
        'prf_size_label2code': {},
        'prf_size_code2mid': {},
        'current_shap_values': None,
        'current_model_explainer': None,
        'last_prediction_inputs': None,
        'current_prediction_results': None
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

# --- Field helper functions using merged config ---
def get_field_label(field_name):
    """Get display label for a field"""
    return FIELDS.get(field_name, {}).get("label", field_name.replace("_", " ").title())

def get_field_title(field_name):
    """Get title for a field"""
    return FIELDS.get(field_name, {}).get("title", get_field_label(field_name))

def get_field_help(field_name):
    """Get help text for a field"""
    return FIELDS.get(field_name, {}).get("help", "")

def get_field_options(field_name):
    opts = None
    raw_opts = None
    # Make sure the field exists in the mapping and is not None
    mapping = CATEGORICAL_MAPPING.get(field_name)
    if not mapping:
        return []

    raw_opts = mapping.get('options')
    if raw_opts is None:
        return []
    
    # Special handling for project_prf_relative_size if it's a dict or list of dicts
    if field_name == "project_prf_relative_size":
        # List of dicts style
        if isinstance(raw_opts, list) and raw_opts and isinstance(raw_opts[0], dict):
            opts = [v['label'] for v in raw_opts]  # Result: ["Small", "Medium"] - shown in dropdown
            # Save mappings in session state for later lookups
            st.session_state.prf_size_label2code = {v['label']: v['code'] for v in raw_opts}  # Result: {"Small": "S", "Medium": "M"}
            st.session_state.prf_size_code2mid = {v['code']: v['midpoint'] for v in raw_opts}  # Result: {"S": 75, "M": 300}
            # Save the entire option by code for future needs (e.g., min/max hour lookups)
            st.session_state.prf_size_code2full = {v.get('code', ''): v for v in raw_opts}  # Result: {"S": {full S dict}, "M": {full M dict}}
            print(f"DEBUG: relative size options = {opts}")
            return opts
        else:
            print(f"DEBUG: relative size options = {raw_opts}")
            return raw_opts  # fallback
    else:
        return raw_opts

def get_tab_organization():
    """Get tab organization from configuration"""
    return UI_INFO_CONFIG.get('tab_organization', {
        "Important Features": [],
        "Nice Features": []
    })

def render_field(field_name, config, is_required=False):
    """Render a form field based on its configuration"""
    label = config.get("label", field_name)
    help_text = config.get("help", "")
    field_type = config.get("type", "text")
    value = config.get("default")
    field_value = None

    if is_required:
        label = f"{label} {UIConstants.REQUIRED_FIELD_MARKER}"

    if field_type == "numeric":
        min_val = config.get("min", 0)
        max_val = config.get("max", 9999)
        field_value = st.number_input(
            label, min_value=min_val, max_value=max_val, value=value, help=help_text, key=field_name
        )
    elif field_type == "categorical":
        options = get_field_options(field_name)
        default = config.get("default", options[0] if options else None)
        
        try:
            default_index = options.index(default)
        except (ValueError, IndexError):
            default_index = 0

        # For project_prf_relative_size, show label, store code in user_inputs
        if field_name == "project_prf_relative_size":
            # Defensive: ensure mapping exists
            if "prf_size_label2code" not in st.session_state:
                get_field_options(field_name)
            field_value_label = st.selectbox(
                label, options,
                index=default_index if options else None,
                help=help_text,
                key=field_name
            )
            # If user picks the empty or None, fallback to None or ""
            field_value = st.session_state.prf_size_label2code.get(field_value_label, None)
        else:
            field_value = st.selectbox(
                label, options,
                index=default_index if options else None,
                help=help_text,
                key=field_name
            )

    elif field_type == "boolean":
        field_value = st.checkbox(label, value=bool(value), help=help_text, key=field_name)
    else:
        field_value = st.text_input(label, value=str(value) if value else "", help=help_text, key=field_name)
    return field_value

# ------------------- SIMPLIFIED SIDEBAR FUNCTION ----------------------

def sidebar_inputs():
    """Create simplified sidebar inputs without save/load functionality"""
    with st.sidebar:
        st.title("Project Parameters")
        st.info("Required fields (marked with ⭐)")
        user_inputs = {}

        # Get tab organization dynamically
        tab_org = get_tab_organization()
        tabs = st.tabs(list(tab_org.keys()))
        for idx, (tab_name, field_list) in enumerate(tab_org.items()):
            with tabs[idx]:
                for field_name in field_list:
                    config = FIELDS.get(field_name)
                    if not config:
                        st.warning(f"Field '{field_name}' not configured.")
                        continue
                    is_required = config.get("mandatory", False)
                    if field_name == "project_prf_functional_size":
                        rel_code = user_inputs.get("project_prf_relative_size")
                        if rel_code and rel_code in st.session_state.prf_size_code2mid:
                            config["default"] = st.session_state.prf_size_code2mid[rel_code]
                        else:
                            config["default"] = config.get("default", 5)
                    field_value = render_field(field_name, config, is_required)
                    user_inputs[field_name] = field_value

        st.divider()
        st.subheader("🤖 Model Selection")
        selected_model = None
        
        try:
            # Use cached model system instead of repeated calls
            if 'cached_model_system' not in st.session_state:
                st.session_state.cached_model_system = initialize_model_system_cached()
            
            model_system = st.session_state.cached_model_system
            
            if model_system.get("initialized") and model_system["status"].get("models_available", False):
                available_models = model_system["models"]
                if available_models:
                    model_options = {m['display_name']: m['technical_name'] for m in available_models}
                    selected_display_name = st.selectbox(
                        "Choose ML Model",
                        list(model_options.keys()),
                        help="Select a model for prediction."
                    )
                    selected_model = model_options[selected_display_name]
                    
                    if st.session_state["prediction_history"]:
                        st.info(f"📊 {len(st.session_state['prediction_history'])} predictions made so far")
                else:
                    st.warning("⚠️ No trained models found")
            else:
                st.warning("⚠️ Models not available")
                if model_system.get('error'):
                    st.error(f"Model loading error: {model_system['error']}")
        except Exception as e:
            st.error(f"Model loading error: {e}")
            selected_model = None

        required_fields = [fname for fname, fdef in FIELDS.items() if fdef.get("mandatory", False)]
        missing_fields = []
        for field in required_fields:
            value = user_inputs.get(field)
            if value is None or value == "" or value == []:
                missing_fields.append(get_field_label(field))

        if missing_fields and st.session_state.get('form_attempted'):
            st.error(f"⚠️ Missing required fields: {', '.join(missing_fields)}")

        st.divider()
        predict_button = st.button(
            "🔮 Predict Effort",
            type="primary",
            use_container_width=True,
            disabled=len(missing_fields) > 0 or not selected_model
        )
        if predict_button:
            st.session_state['form_attempted'] = True

        st.subheader("📈 Prediction History")
        col1, col2 = st.columns(2)
        with col1:
            clear_results = st.button(
                "🗑️ Clear History",
                use_container_width=True,
                help="Clear all previous predictions"
            )
        with col2:
            show_history = st.button(
                "📊 Show All",
                use_container_width=True,
                help="Show detailed prediction history"
            )

        # Handle clear button
        if clear_results:
            st.session_state['prediction_history'] = []
            st.session_state['current_prediction_results'] = None
            st.rerun()

        # Handle show all button
        if show_history:
            st.session_state['show_detailed_history'] = True    
                
        user_inputs["selected_model"] = selected_model
        user_inputs["submit"] = predict_button
        user_inputs["show_history"] = show_history

        return user_inputs

# --- Display Functions ---
def display_inputs(user_inputs, selected_model):
    """Display input parameters summary in a collapsible expander"""
    with st.expander("📋Input Parameters Summary", expanded=False):
        exclude_keys = {'selected_model', 'submit'}
        items = [(get_field_label(k), v) for k, v in user_inputs.items() 
                if k not in exclude_keys and v is not None and v != ""]
        
        if items:
            # Group items for better display
            col1, col2 = st.columns(2)
            mid = len(items) // UIConstants.COLUMN_SPLIT_RATIO
            
            with col1:
                for param, value in items[:mid]:
                    st.text(f"**{param}:** {value}")
            with col2:
                for param, value in items[mid:]:
                    st.text(f"**{param}:** {value}")
            
            # Show selected model
            if selected_model:
                try:
                    model_display_name = get_model_display_name(selected_model)
                    st.info(f"**Model:** {model_display_name}")
                except:
                    st.info(f"🤖 **Model:** {selected_model}")
        else:
            st.warning("No parameters to display")

def show_prediction(prediction, model_name, user_inputs=None):
    """Show prediction results with team breakdown and dynamic size-band warnings."""
    if prediction is None:
        st.error("Prediction failed. Please check your inputs and try again.")
        return
    
    st.subheader("🎯 Prediction Results")
    
    try:
        model_display_name = get_model_display_name(model_name)
        st.info(f"**Model Used:** {model_display_name}")
    except:
        st.info(f"**Model Used:** {model_name}")
    
    # Main prediction metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(UIConstants.EFFORT_METRIC_TEMPLATE, f"{prediction:.0f} hours")

    with col2:
        days = prediction / UIConstants.HOURS_PER_DAY
        st.metric(UIConstants.DAYS_METRIC_TEMPLATE, f"{days:.1f} days")

    with col3:
        weeks = days / UIConstants.DAYS_PER_WEEK
        st.metric(UIConstants.WEEKS_METRIC_TEMPLATE, f"{weeks:.1f} weeks")

    with col4:
        months = weeks / UIConstants.WEEKS_PER_MONTH
        st.metric(UIConstants.MONTHS_METRIC_TEMPLATE, f"{months:.1f} months")

    # Dynamic thresholds based on selected relative size
    if user_inputs is not None and "project_prf_relative_size" in user_inputs:
        rel_code = user_inputs["project_prf_relative_size"]
        size_info = st.session_state["prf_size_code2full"].get(rel_code, {})
        min_hour = size_info.get("minimumhour", 0)
        max_hour = size_info.get("maximumhour", None)

        if prediction < min_hour:
            st.warning(
                f"⚠️ The prediction ({prediction:.0f} hours) is **below** the expected minimum for this size ({min_hour} hours). "
                "Please check your inputs."
            )
        elif max_hour and prediction > max_hour:
            st.warning(
                f"⚠️ The prediction ({prediction:.0f} hours) is **above** the expected maximum for this size ({max_hour} hours). "
                "Consider breaking down the project or reviewing your parameters."
            )

def show_feature_importance(model_name, features_dict):
    """Display feature importance analysis"""
    try:
        feature_importance = get_feature_importance(model_name)
        if feature_importance is None:
            st.info("Feature importance analysis not available for this model.")
            return
        
        # Get display name for the model
        try:
            model_display_name = get_model_display_name(model_name)
        except Exception:
            model_display_name = model_name

        st.subheader(f"📊 Feature Importance Analysis (Model: {model_display_name})")

        
        exclude_keys = {'selected_models', 'submit', 'clear_results', 'comparison_mode', 'selected_model', 'show_history'}
        feature_names = [k for k in features_dict.keys() if k not in exclude_keys]
        
        # Get display settings
        max_features = FEATURE_IMPORTANCE_DISPLAY.get('max_features_shown', 15)
        precision = FEATURE_IMPORTANCE_DISPLAY.get('precision_decimals', 3)
        
        importance_data = []
        for i, name in enumerate(feature_names[:max_features]):
            if i < len(feature_importance):
                friendly_name = get_field_title(name)
                importance_data.append({
                    'Feature': friendly_name,
                    'Importance': abs(feature_importance[i])
                })
        
        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Display chart
            st.bar_chart(importance_df.set_index('Feature'))
            
            with st.expander("View Detailed Importance Data"):
                st.dataframe(importance_df.round(precision), use_container_width=True)
        else:
            st.warning("No feature importance data available")
    
    except Exception as e:
        st.info(f"Feature importance analysis not available: {e}")

def display_instance_specific_shap(user_inputs, model_name):
    """Display SHAP analysis using UI configuration for proper field names"""
    st.subheader("Feature Analysis")
    
    # Always show fallback analysis first
    display_fallback_analysis(user_inputs, model_name)
    
    # Make SHAP optional with button
    if st.button("Calculate Advanced SHAP", key="calc_shap"):
        with st.spinner("Calculating SHAP values..."):
            from agileee.shap_analysis import get_shap_analysis_results
            
            results = get_shap_analysis_results(user_inputs, model_name, get_trained_model)
            
            if results.get('error'):
                st.warning(f"SHAP analysis failed: {results['error']}")
            elif results.get('success'):
                display_shap_results_ui(results['shap_values'], user_inputs, model_name)

def display_fallback_analysis(user_inputs, model_name):
    """Show fallback analysis using proper UI field names"""
    st.markdown("### Your Project Configuration")
    for key, value in user_inputs.items():
        if key not in {'selected_model', 'submit'}:
            clean_key = get_field_label(key)  # Uses FIELDS config
            st.write(f"**{clean_key}:** {value}")
    
    st.markdown("### General Feature Importance")
    st.markdown("""
    - **Project Size**: Usually the strongest predictor of effort
    - **Team Size**: Moderate to strong impact on total effort  
    - **Technology Complexity**: Can significantly affect development time
    - **Industry/Domain**: Influences requirements complexity
    """)

def display_shap_results_ui(shap_values, user_inputs, model_name):
    """Display SHAP results using UI configuration"""
    st.subheader("SHAP Feature Contributions")
    
    # Get top features using proper display names
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[::-1][:PipelineConstants.TOP_N_FEATURES]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top Effort Increasers**")
        for idx in top_indices:
            if shap_values[idx] > 0:
                # Get proper field name from FIELDS config
                field_names = list(user_inputs.keys())
                if idx < len(field_names):
                    field_key = field_names[idx]
                    display_name = get_field_label(field_key)
                    st.write(f"• **{display_name}**: +{shap_values[idx]:.4f}")
    
    with col2:
        st.markdown("**Top Effort Reducers**")  
        for idx in top_indices:
            if shap_values[idx] < 0:
                field_names = list(user_inputs.keys())
                if idx < len(field_names):
                    field_key = field_names[idx]
                    display_name = get_field_label(field_key)
                    st.write(f"• **{display_name}**: {shap_values[idx]:.4f}")

def show_prediction_history():
    """Display prediction history"""
    if not st.session_state['prediction_history']:
        return
    
    st.subheader("📈 Prediction History")
    
    history_data = []
    
    try:
        for entry in st.session_state["prediction_history"]:
            # Safely extract model information
            model_technical = entry.get('model_technical', '')
            if not model_technical:
                model_technical = entry.get('model', 'Unknown Model')
            
            # Get display name with fallback
            model_display = entry.get('model', model_technical)
            if not model_display:
                model_display = model_technical

            # Build history entry
            history_entry = {
                'Timestamp': entry.get('timestamp', 'Unknown'),
                'Model': model_display,
                'Hours': f"{entry.get('prediction_hours', 0):.0f}",
                'Days': f"{entry.get('prediction_hours', 0)/UIConstants.HOURS_PER_DAY:.1f}"
            }
            history_data.append(history_entry)
        
        # Display the data
        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No prediction history to display")
            
    except Exception as e:
        st.error(f"Error displaying prediction history: {str(e)}")
def show_prediction_comparison_table():
    """Show comparison table if multiple predictions exist"""
    if len(st.session_state["prediction_history"]) <= 1:
        return
    
    st.subheader("🔍 Prediction Comparison")
    
    try:
        predictions = []
        models = []
        
        for entry in st.session_state["prediction_history"]:
            # Extract prediction safely
            prediction_hours = entry.get('prediction_hours', 0)
            predictions.append(prediction_hours)
            
            # Extract model name safely
            model_technical = entry.get('model_technical', '')
            if not model_technical:
                model_technical = entry.get('model', 'Unknown Model')
            
            # Get display name with fallback
            try:
                from models import get_model_display_name_from_config
                model_display = get_model_display_name_from_config(model_technical)
            except Exception:
                model_display = entry.get('model', model_technical)
                if not model_display:
                    model_display = model_technical
            
            models.append(model_display)
        
        # Create comparison data
        comparison_data = {
            'Model': models,
            'Hours': predictions,
            'Days': [p/UIConstants.HOURS_PER_DAY for p in predictions]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Statistics
        if len(predictions) > 1:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average", f"{np.mean(predictions):.0f} hours")
            with col2:
                st.metric("Min", f"{np.min(predictions):.0f} hours")
            with col3:
                st.metric("Max", f"{np.max(predictions):.0f} hours")
            with col4:
                st.metric("Std Dev", f"{np.std(predictions):.0f} hours")
                
    except Exception as e:
        st.error(f"Error creating comparison table: {str(e)}")

def add_prediction_to_history(user_inputs, model_name, prediction):
    """Add prediction to session history"""
    if prediction is None:
        return
    
    try:
        # Get display name safely
        try:
            from models import get_model_display_name_from_config
            model_display_name = get_model_display_name_from_config(model_name)
        except Exception:
            model_display_name = model_name
        
        history_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': model_display_name,
            'model_technical': model_name,
            'prediction_hours': prediction,
            'inputs': user_inputs.copy() if user_inputs else {}
        }
        
        st.session_state["prediction_history"].append(history_entry)
        
    except Exception as e:
        st.error(f"Error adding prediction to history: {str(e)}")

def display_model_comparison():
    """Display model comparison analysis"""
    st.header("🤖 Model Comparison")
    
    if len(st.session_state["prediction_history"]) < 2:
        st.warning("⚠️ Please make predictions with at least 2 different models to enable comparison.")
        return
    
    try:
        # Group predictions by model
        model_predictions = {}
        
        for entry in st.session_state["prediction_history"]:
            # Use technical name for grouping to avoid display name inconsistencies
            model_name = entry.get('model_technical', entry.get('model', 'Unknown'))
            prediction_hours = entry.get('prediction_hours', 0)
            
            if model_name not in model_predictions:
                model_predictions[model_name] = []
            model_predictions[model_name].append(prediction_hours)
        
        # Create comparison visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Performance Comparison")
            
            # Box plot for model predictions
            comparison_data = []
            for model, predictions in model_predictions.items():
                # Get display name for visualization
                try:
                    from models import get_model_display_name_from_config
                    display_name = get_model_display_name_from_config(model)
                except Exception:
                    display_name = model
                
                for pred in predictions:
                    comparison_data.append({
                        'Model': display_name,
                        'Prediction (Hours)': pred
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                # Create box plot using plotly
                fig = px.box(data_frame=comparison_df, x='Model', y='Prediction (Hours)',
                            title="Distribution of Predictions by Model")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Model Statistics")
            
            # Statistics table
            stats_data = []
            for model, predictions in model_predictions.items():
                if predictions:
                    try:
                        from models import get_model_display_name_from_config
                        display_name = get_model_display_name_from_config(model)
                    except Exception:
                        display_name = model
                    
                    stats_data.append({
                        'Model': display_name,
                        'Count': len(predictions),
                        'Mean': f"{np.mean(predictions):.0f}",
                        'Std Dev': f"{np.std(predictions):.0f}",
                        'Min': f"{np.min(predictions):.0f}",
                        'Max': f"{np.max(predictions):.0f}"
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in model comparison: {str(e)}")

def display_static_shap_analysis():
    """Display static SHAP analysis from file"""
    st.header("📈 Static SHAP Analysis - Model Feature Importance")

    try:
        with open(FileConstants.SHAP_ANALYSIS_FILE, "r", encoding="utf-8") as f:
            shap_report_md = f.read()
        st.markdown(shap_report_md, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load static SHAP analysis report: {e}")

def about_section():
    """Display about section with tool information"""
    st.markdown("""
    ### About This Tool
    
    The **ML Project Effort Estimator** is a machine learning-powered tool designed to help project managers, 
    developers, and teams estimate the effort required for software development projects.
    
    #### Key Features:
    - **ML Models**: Predictions from trained machine learning models
    - **Comprehensive Parameters**: Considers project size, team composition, technology stack, and organizational factors
    - **Interactive Interface**: User-friendly form with real-time validation and feedback
    - **SHAP Analysis**: Insights into feature importance and model behavior
    - **Historical Tracking**: Keep track of previous predictions for comparison
    
    #### How It Works:
    1. **Data Input**: Enter project parameters including team size, technology stack, and project characteristics
    2. **Model Selection**: Choose an ML model for prediction
    3. **ML Prediction**: The tool applies trained machine learning models to generate effort estimates
    4. **Results Analysis**: View the predicted effort in hours, days, and breakdowns
    5. **SHAP Analysis**: Understand which factors most influence your project's effort estimate
    
    #### Best Practices:
    - Provide accurate team size and project complexity information
    - Use the SHAP analysis to understand model behavior
    - Consider the tool's predictions as guidance alongside expert judgment
    
    For technical support or questions, please refer to the documentation or contact the development team.
    """)

# --- Main Application Function ---
def main():
    """Main application function with simplified interface"""

    # Initialize configs once
    initialize_configs_once()
    
    # Initialize session state
    initialize_session_state()
    
    # Set sidebar width
    set_sidebar_width()
    
    # Main header
    st.title("🔮 ML Agile Software Project Effort Estimator")
    st.markdown("Get accurate effort estimates using machine learning models trained on historical project data.")
    
   
    try:
        # Get user inputs from sidebar
        user_inputs = sidebar_inputs()
        
        # --- Tab navigation for main content ---
        main_tabs = st.tabs(["🔮 Estimator", "📊 Visualisations & Analysis", "🤖 Model Comparison", "📈 Static SHAP Analysis", "❓ Help"])

        with main_tabs[0]:  # Estimator tab

            # Check if show_history button was clicked
            if user_inputs.get('show_history', False):
                st.header("📊 Detailed Prediction History")
                show_prediction_history()
                show_prediction_comparison_table()
                return  # Exit early to show only history
    
            if user_inputs.get('submit', False):
                selected_model = user_inputs.get('selected_model')
                
                if selected_model:
                    # Run prediction and store in session state
                    with st.spinner("Calculating estimation..."):
                        try:
                            prediction = predict_man_hours(user_inputs, selected_model)
                            
                            # Store complete results in session state
                            st.session_state['current_prediction_results'] = {
                                'prediction': prediction,
                                'model': selected_model,
                                'inputs': user_inputs.copy()
                            }
                            
                            add_prediction_to_history(user_inputs, selected_model, prediction)
                            
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
                            st.session_state['current_prediction_results'] = None
                else:
                    st.warning("Please select a model to make predictions")
            
            # Display results from session state (persists across model changes)
            if st.session_state.get('current_prediction_results'):
                results = st.session_state['current_prediction_results']
                display_inputs(results['inputs'], results['model'])
                st.divider()
                show_prediction(results['prediction'], results['model'], results['inputs'])
                show_prediction_history()
                st.divider()
                show_feature_importance(results['model'], results['inputs'])
            else:
                # Welcome screen
                st.info("**Get Started:** Fill in the project parameters in the sidebar and click 'Predict Effort' to get your estimate.")

        with main_tabs[1]:  # Instance-Specific SHAP tab
            st.header("Instance-Specific SHAP Analysis")
            
            # Check if we have a recent prediction to analyze
            if not st.session_state["prediction_history"]:
                st.warning("Please make at least one prediction first to enable SHAP analysis.")
            else:
                # Get the most recent prediction for analysis
                latest_prediction = st.session_state["prediction_history"][-1]
                user_inputs = latest_prediction.get('inputs', {})
                model_name = latest_prediction.get('model_technical')
                
                if not user_inputs or not model_name:
                    st.error("Cannot perform analysis - missing prediction data.")
                else:
                    st.info("Enhanced SHAP Analysis: Using optimized feature analysis for faster performance with high accuracy.")
                    display_instance_specific_shap(user_inputs, model_name)

        with main_tabs[2]:  # Model Comparison tab
            display_model_comparison()

        with main_tabs[3]:  # Static SHAP Analysis tab
            display_static_shap_analysis()

        with main_tabs[4]:  # Help tab            
            with st.expander("How to Use This Tool"):
                st.markdown(f"""
                ### Quick Start Guide
                
                1. **Fill Required Fields** - Complete all fields marked with {UIConstants.REQUIRED_FIELD_MARKER} in the sidebar
                2. **Optional Parameters** - Add more details for better accuracy  
                3. **Select Model** - Choose a model for prediction
                4. **Get Prediction** - Click 'Predict Effort' to see your estimate
                5. **Analyze Results** - Use the Instance-Specific SHAP tab for insights
                
                ### Features
                - **Detailed Predictions**: Hours, days, and breakdowns
                - **Prediction History**: Track and compare multiple predictions
                - **SHAP Analysis**: Understanding of feature importance
                - **Model Comparison**: Analyze different models
                
                ### Tips for Better Estimates
                - Fill in as many relevant fields as possible
                - Use realistic team sizes and project characteristics
                - Explore the SHAP analysis after making predictions
                - Consider the tool's predictions as guidance alongside expert judgment
                """)
            
            # About section
            with st.expander("About This Tool"):
                about_section()
    
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your configuration files and model setup.")

if __name__ == "__main__":
    main()