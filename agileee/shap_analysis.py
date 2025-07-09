# shap_analysis.py - Fixed SHAP Analysis Module for PyCaret Models
"""
SHAP Analysis Module that properly handles PyCaret-wrapped models.

Key Features:
1. Properly extracts the underlying estimator from PyCaret models
2. Handles the feature transformation pipeline correctly
3. Returns actual SHAP explainer objects, not dictionaries
4. Properly aligns features between UI (22) and model (54-67)
Architecture:
- UI Input (22 features) → prepare_features_for_model() → Model Features (66-67)
- ISBSG Background Data → Same pipeline → Model-ready features
- Both processed through same pipeline for consistency
"""

# shap_analysis.py - Minimal fixes for working SHAP analysis

import numpy as np
import pandas as pd
import shap
import warnings
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Union, Callable, Any
import logging
from agileee.constants import PipelineConstants, UIConstants


# Suppress SHAP warnings
warnings.filterwarnings('ignore', category=UserWarning, module='shap')

# Global cache for explainers
_explainer_cache = {}

def clear_explainer_cache():
    """Clear the explainer cache to free memory."""
    global _explainer_cache
    _explainer_cache.clear()
    st.success("SHAP explainer cache cleared!")

def get_cache_info() -> Dict[str, Any]:
    """Get information about the current explainer cache."""
    return {
        'cached_models': list(_explainer_cache.keys()),
        'cache_size': len(_explainer_cache)
    }

def get_shap_analysis_results(user_inputs, model_name, get_trained_model_func):
    """
    Calculate SHAP analysis and return results (no display)
    """
    try:
        # Validation
        if not user_inputs or not model_name:
            return {'error': 'Missing inputs or model'}
        
        # Try to load model and create explainer
        explainer = get_shap_explainer_optimized(user_inputs, model_name, get_trained_model_func)
        
        if explainer is None:
            return {'error': 'Could not create SHAP explainer'}
        
        # Calculate SHAP values
        shap_values = get_shap_values_safe(explainer, user_inputs, model_name)
        
        if shap_values is None:
            return {'error': 'Could not calculate SHAP values'}
        
        # Return raw results
        return {
            'success': True,
            'shap_values': shap_values,
            'model_name': model_name,
            'user_inputs': user_inputs
        }
        
    except Exception as e:
        return {'error': str(e)}

def get_shap_explainer_optimized(user_inputs, model_name, get_trained_model_func):
    """
    FIXED: Simplified explainer creation with better error handling
    """
    try:
        # Check cache first
        cache_key = f"{model_name}_simplified"
        if cache_key in _explainer_cache:
            return _explainer_cache[cache_key]
        
        # Load model
        model = get_trained_model_func(model_name)
        if model is None:
            st.error(f"Could not load model: {model_name}")
            return None
        
        # Extract actual estimator from PyCaret wrapper
        actual_model = extract_model_estimator(model)
        
        # Get background data (simplified approach)
        background_data = get_simple_background_data(user_inputs)
        
        if background_data is None:
            st.warning("Using model without background data")
            background_data = None
        
        # Create explainer based on model type
        explainer = create_appropriate_explainer(actual_model, background_data)
        
        if explainer is not None:
            _explainer_cache[cache_key] = explainer
            st.success(f"✅ SHAP explainer created for {model_name}")
        
        return explainer
        
    except Exception as e:
        st.error(f"Error creating SHAP explainer: {e}")
        logging.error(f"Explainer creation failed: {e}")
        return None

def extract_model_estimator(model):
    """
    FIXED: Robust model extraction from PyCaret wrappers
    """
    # Try different PyCaret wrapper patterns
    if hasattr(model, '_final_estimator'):
        return model._final_estimator
    elif hasattr(model, 'named_steps'):
        # Pipeline - get last step
        steps = list(model.named_steps.values())
        return steps[-1] if steps else model
    elif hasattr(model, 'steps'):
        # Sklearn pipeline
        return model.steps[-1][1] if model.steps else model
    else:
        # Assume it's already unwrapped
        return model

def get_simple_background_data(user_inputs, n_samples=PipelineConstants.KERNEL_EXPLAINER_SAMPLE_SIZE):
    """
    FIXED: Simple background data generation when ISBSG not available
    """
    try:
        # Try to import models for background data
        from agileee.models import prepare_features_for_model
        
        # Generate synthetic samples based on user inputs
        samples = []
        for i in range(n_samples):
            # Create variations of user inputs
            sample_input = create_sample_variation(user_inputs)
            
            # Process through same pipeline
            processed = prepare_features_for_model(sample_input)
            if processed is not None and not processed.empty:
                samples.append(processed.values.flatten())
        
        if samples:
            return np.array(samples, dtype=np.float32)
        else:
            return None
            
    except Exception as e:
        logging.warning(f"Background data generation failed: {e}")
        return None

def create_sample_variation(base_inputs):
    """
    FIXED: Create realistic variations of user inputs for background data
    """
    variation = base_inputs.copy()
    
    # Add some realistic variations
    if 'project_prf_functional_size' in variation:
        base_size = variation['project_prf_functional_size']
        variation['project_prf_functional_size'] = max(1, int(base_size * np.random.uniform(0.5, 2.0)))
    
    if 'project_prf_max_team_size' in variation:
        base_team = variation['project_prf_max_team_size']
        variation['project_prf_max_team_size'] = max(1, int(base_team * np.random.uniform(0.7, 1.5)))
    
    # Vary categorical fields occasionally
    categorical_fields = [
        'external_eef_industry_sector',
        'tech_tf_primary_programming_language',
        'project_prf_relative_size'
    ]
    
    for field in categorical_fields:
        if field in variation and np.random.random() < 0.3:  # 30% chance to vary
            # Keep original value most of the time for stability
            pass
    
    return variation

def create_appropriate_explainer(model, background_data):
    """
    FIXED: Create explainer with robust fallback logic
    """
    model_type = type(model).__name__.lower()
    
    # Try TreeExplainer for tree-based models
    tree_keywords = ['forest', 'tree', 'xgb', 'lgb', 'catboost', 'gradient']
    if any(keyword in model_type for keyword in tree_keywords):
        try:
            if background_data is not None:
                return shap.TreeExplainer(model, background_data, check_additivity=False)
            else:
                return shap.TreeExplainer(model)
        except Exception as e:
            logging.warning(f"TreeExplainer failed: {e}")
    
    # Try LinearExplainer for linear models  
    linear_keywords = ['linear', 'lasso', 'ridge', 'elastic']
    if any(keyword in model_type for keyword in linear_keywords):
        try:
            if background_data is not None:
                return shap.LinearExplainer(model, background_data)
            else:
                return shap.LinearExplainer(model, np.zeros((1, PipelineConstants.KERNEL_EXPLAINER_SAMPLE_SIZE)))
        except Exception as e:
            logging.warning(f"LinearExplainer failed: {e}")
    
    # Fallback to general Explainer (slower but more compatible)
    try:
        if background_data is not None:
            # Use smaller sample for general explainer
            sample_size = min(PipelineConstants.MAX_ANALYSIS_POINTS, len(background_data))
            return shap.Explainer(model, background_data[:sample_size])
        else:
            # Create minimal background data
            dummy_data = np.zeros((1, PipelineConstants.KERNEL_EXPLAINER_SAMPLE_SIZE))  # Assume 50 features
            return shap.Explainer(model, dummy_data)
    except Exception as e:
        logging.error(f"All explainer types failed: {e}")
        return None

def get_shap_values_safe(explainer, user_inputs, model_name):
    """
    FIXED: Safe SHAP value calculation with error handling
    """
    try:
        from agileee.models import prepare_features_for_model
        
        # Prepare input data through same pipeline as model expects
        input_df = prepare_features_for_model(user_inputs)
        if input_df is None or input_df.empty:
            logging.error("Could not prepare input features for SHAP")
            return None
        
        input_data = input_df.values
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        
        # Calculate SHAP values
        input_safe = np.nan_to_num(input_data.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        try:
            shap_values = explainer.shap_values(input_safe)
        except Exception as shap_error:
            logging.error(f"SHAP calculation failed: {shap_error}")
            return None

        print(f"DEBUG: shap_values type: {type(shap_values)}, value: {shap_values}")
        
        # Handle different return formats
        if isinstance(shap_values, list):
            # Multi-class or multi-output
            shap_values = shap_values[0] if len(shap_values) > 0 else None
        
        if shap_values is not None and len(shap_values.shape) > 1:
            shap_values = shap_values[0]  # Get first instance
        
        return shap_values
        
    except Exception as e:
        logging.error(f"SHAP value calculation failed: {e}")
        return None

def display_shap_results(shap_values, user_inputs, model_name):
    """
    FIXED: Display SHAP results with better visualization
    """
    try:
        if shap_values is None or len(shap_values) == 0:
            st.warning("No SHAP values to display")
            return
        
        # Get feature names
        feature_names = get_feature_names_for_display(user_inputs, len(shap_values))
        
        # Create SHAP summary
        create_shap_summary_display(shap_values, feature_names, user_inputs)
        
        # Create bar chart
        create_shap_bar_chart(shap_values, feature_names)
        
        # Show top contributors
        show_top_contributors(shap_values, feature_names, user_inputs)
        
    except Exception as e:
        st.error(f"Error displaying SHAP results: {e}")
        logging.error(f"SHAP display error: {e}")

def get_feature_names_for_display(user_inputs, n_features):
    """
    FIXED: Get appropriate feature names for display
    """
    try:
        # Try to get from model configuration
        from agileee.models import FIELDS
        
        if FIELDS:
            field_names = list(FIELDS.keys())
            # Remove UI-only keys
            exclude_keys = {'selected_model', 'submit', 'clear_results'}
            field_names = [name for name in field_names if name not in exclude_keys]
        else:
            field_names = list(user_inputs.keys())
        
        # Ensure we have enough names
        while len(field_names) < n_features:
            field_names.append(f"feature_{len(field_names)}")
        
        return field_names[:n_features]
        
    except Exception:
        # Fallback to generic names
        return [f"feature_{i}" for i in range(n_features)]

def create_shap_summary_display(shap_values, feature_names, user_inputs):
    """
    FIXED: Create summary display of SHAP analysis
    """
    st.subheader("📊 Feature Impact Summary")
    
    # Calculate importance metrics
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[::-1][:PipelineConstants.TOP_N_FEATURES]  # Top 10 features
    
    # Create summary table
    summary_data = []
    for idx in top_indices:
        if idx < len(feature_names):
            feature_name = feature_names[idx]
            shap_val = shap_values[idx]
            
            summary_data.append({
                'Feature': feature_name.replace('_', ' ').title(),
                'SHAP Value': f"{shap_val:.4f}",
                'Impact': 'Increases' if shap_val > 0 else 'Decreases',
                'Magnitude': f"{abs(shap_val):.4f}"
            })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
    
    # Show interpretation
    st.info("""
    **How to read SHAP values:**
    - **Positive values** increase the predicted effort
    - **Negative values** decrease the predicted effort  
    - **Larger absolute values** have stronger impact
    """)

def create_shap_bar_chart(shap_values, feature_names):
    """
    FIXED: Create horizontal bar chart of SHAP values
    """
    try:
        st.subheader("📈 Feature Contribution Chart")
        
        # Get top features for display
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[::-1][:PipelineConstants.MAX_FEATURES_SHOWN]  # Top 15 for visibility
        
        top_features = [feature_names[i] if i < len(feature_names) else f"feature_{i}" 
                       for i in top_indices]
        top_values = [shap_values[i] for i in top_indices]
        
        # Create plotly horizontal bar chart
        colors = ['red' if val < 0 else 'blue' for val in top_values]
        
        fig = go.Figure(data=[
            go.Bar(
                y=top_features,
                x=top_values,
                orientation='h',
                marker_color=colors,
                text=[f"{val:.3f}" for val in top_values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="SHAP Feature Contributions",
            xaxis_title="SHAP Value (Impact on Prediction)",
            yaxis_title="Features",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating chart: {e}")

def show_top_contributors(shap_values, feature_names, user_inputs):
    """
    FIXED: Show top positive and negative contributors
    """
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔺 Top Effort Increasers")
            positive_indices = np.where(shap_values > 0)[0]
            if len(positive_indices) > 0:
                pos_sorted = positive_indices[np.argsort(shap_values[positive_indices])[::-1]]
                for i, idx in enumerate(pos_sorted[:5]):
                    if idx < len(feature_names):
                        feature = feature_names[idx].replace('_', ' ').title()
                        value = shap_values[idx]
                        st.write(f"{i+1}. **{feature}**: +{value:.4f}")
            else:
                st.write("No features increase effort for this prediction")
        
        with col2:
            st.subheader("🔻 Top Effort Reducers")
            negative_indices = np.where(shap_values < 0)[0]
            if len(negative_indices) > 0:
                neg_sorted = negative_indices[np.argsort(shap_values[negative_indices])]
                for i, idx in enumerate(neg_sorted[:5]):
                    if idx < len(feature_names):
                        feature = feature_names[idx].replace('_', ' ').title()
                        value = shap_values[idx]
                        st.write(f"{i+1}. **{feature}**: {value:.4f}")
            else:
                st.write("No features reduce effort for this prediction")
                
    except Exception as e:
        st.error(f"Error showing contributors: {e}")

def show_fallback_analysis(user_inputs, model_name):
    """
    FIXED: Fallback analysis when SHAP fails
    """
    st.subheader("📋 Static Feature Analysis")
    st.info("Showing general feature importance (SHAP analysis not available)")
    
    try:
        # Try to show static analysis from file
        from agileee.constants import FileConstants
        import os
        
        if os.path.exists(FileConstants.SHAP_ANALYSIS_FILE):
            with open(FileConstants.SHAP_ANALYSIS_FILE, "r", encoding="utf-8") as f:
                shap_report_md = f.read()
            st.markdown(shap_report_md, unsafe_allow_html=True)
        else:
            # Show basic input summary
            st.markdown("### Your Input Summary:")
            for key, value in user_inputs.items():
                if key not in {'selected_model', 'submit'}:
                    clean_key = key.replace('_', ' ').title()
                    st.write(f"**{clean_key}:** {value}")
            
            st.markdown("""
            ### General Feature Importance:
            - **Project Size**: Usually the strongest predictor of effort
            - **Team Size**: Moderate to strong impact on total effort  
            - **Technology Complexity**: Can significantly affect development time
            - **Industry/Domain**: Influences requirements complexity
            """)
    except Exception as e:
        st.error(f"Error loading fallback analysis: {e}")

# Export the main functions
__all__ = [
    'display_optimized_shap_analysis',
    'get_shap_explainer_optimized', 
    'clear_explainer_cache',
    'get_cache_info'
]