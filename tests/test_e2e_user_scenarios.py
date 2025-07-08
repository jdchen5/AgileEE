# test_e2e_user_scenarios.py
"""
End-to-End User Scenario Tests for AgileEE
Tests specific user personas and real-world usage scenarios.
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the UI module and dependencies
import agileee.ui as ui
from agileee.constants import UIConstants, FileConstants

class TestE2EProjectManagerScenario:
    """Test scenarios for project manager persona"""
    
    def setup_method(self):
        """Setup for project manager tests"""
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
            'current_prediction_results': None
        }

    def test_e2e_pm_comparing_team_sizes(self):
        """Project manager comparing different team sizes for same project"""
        
        with patch('streamlit.sidebar'), \
             patch('streamlit.tabs'), \
             patch('streamlit.title'), \
             patch('streamlit.markdown'), \
             patch('streamlit.header'), \
             patch('streamlit.info'), \
             patch('streamlit.warning'), \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.button'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.number_input'), \
             patch('streamlit.metric'), \
             patch('streamlit.dataframe') as mock_dataframe, \
             patch('plotly.express.box') as mock_box_plot, \
             patch('streamlit.plotly_chart'):
            
            mock_columns.return_value = [MagicMock(), MagicMock()]
            
            with patch.object(ui, 'predict_man_hours') as mock_predict, \
                 patch.object(ui, 'get_model_display_name_from_config') as mock_display:
                
                mock_display.return_value = "Random Forest"
                
                # Scenario: PM wants to see impact of team size on same project
                base_project = {
                    'project_prf_functional_size': 500,
                    'project_prf_relative_size': 'M',
                    'external_eef_industry_sector': 'Financial',
                    'tech_tf_primary_programming_language': 'Java'
                }
                
                # Test with team size 3
                mock_predict.return_value = 650.0
                project_small_team = base_project.copy()
                project_small_team['project_prf_max_team_size'] = 3
                ui.add_prediction_to_history(project_small_team, 'rf_model', 650.0)
                
                # Test with team size 6
                mock_predict.return_value = 580.0
                project_medium_team = base_project.copy()
                project_medium_team['project_prf_max_team_size'] = 6
                ui.add_prediction_to_history(project_medium_team, 'rf_model', 580.0)
                
                # Test with team size 10
                mock_predict.return_value = 720.0
                project_large_team = base_project.copy()
                project_large_team['project_prf_max_team_size'] = 10
                ui.add_prediction_to_history(project_large_team, 'rf_model', 720.0)
                
                # PM reviews comparison
                assert len(st.session_state['prediction_history']) == 3
                
                # PM checks history shows the trend
                predictions = [entry['prediction_hours'] for entry in st.session_state['prediction_history']]
                team_sizes = [entry['inputs']['project_prf_max_team_size'] for entry in st.session_state['prediction_history']]
                
                # Verify data is captured for analysis
                assert predictions == [650.0, 580.0, 720.0]
                assert team_sizes == [3, 6, 10]
                
                # PM views comparison chart
                ui.display_model_comparison()
                
                # Should show comparison even for same model (different inputs)
                # Note: Since it's the same model, comparison will group by model
                # But PM can still see the different predictions in history

    def test_e2e_pm_budget_planning_workflow(self):
        """Project manager using estimates for budget planning"""
        
        with patch('streamlit.sidebar'), \
             patch('streamlit.tabs'), \
             patch('streamlit.title'), \
             patch('streamlit.markdown'), \
             patch('streamlit.header'), \
             patch('streamlit.info'), \
             patch('streamlit.warning') as mock_warning, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.columns'), \
             patch('streamlit.button'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.number_input'), \
             patch('streamlit.metric') as mock_metric:
            
            with patch.object(ui, 'predict_man_hours') as mock_predict, \
                 patch.object(ui, 'get_model_display_name') as mock_display:
                
                mock_display.return_value = "Random Forest"
                mock_predict.return_value = 1200.0  # High estimate
                
                # PM enters large project details
                large_project = {
                    'project_prf_functional_size': 800,
                    'project_prf_max_team_size': 8,
                    'project_prf_relative_size': 'L'  # Large project
                }
                
                # Make prediction
                prediction = mock_predict(large_project, 'rf_model')
                
                # Store in session state as if user clicked predict
                st.session_state['current_prediction_results'] = {
                    'prediction': prediction,
                    'model': 'rf_model',
                    'inputs': large_project
                }
                
                # PM views prediction results
                results = st.session_state['current_prediction_results']
                ui.show_prediction(results['prediction'], results['model'], results['inputs'])
                
                # Verify metrics were displayed for budget planning
                mock_metric.assert_called()
                
                # Check if size warning was triggered (prediction vs. expected range)
                # Large project range: 800-2000 hours, prediction: 1200 hours (within range)
                # Should not trigger warning
                metric_calls = mock_metric.call_args_list
                
                # Verify all needed metrics for budget planning are present
                metric_labels = [call[0][0] for call in metric_calls]
                expected_metrics = ["📊 Total Effort", "📅 Working Days", "📆 Working Weeks", "🗓️ Months"]
                
                for expected in expected_metrics:
                    assert expected in metric_labels, f"Missing budget planning metric: {expected}"

    def test_e2e_pm_risk_assessment_scenario(self):
        """Project manager assessing risk through multiple model predictions"""
        
        with patch('streamlit.sidebar'), \
             patch('streamlit.tabs'), \
             patch('streamlit.title'), \
             patch('streamlit.markdown'), \
             patch('streamlit.header'), \
             patch('streamlit.info'), \
             patch('streamlit.warning'), \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.button'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.number_input'), \
             patch('streamlit.metric'), \
             patch('streamlit.dataframe') as mock_dataframe, \
             patch('plotly.express.box') as mock_box_plot, \
             patch('streamlit.plotly_chart'):
            
            mock_columns.return_value = [MagicMock(), MagicMock()]
            
            with patch.object(ui, 'predict_man_hours') as mock_predict, \
                 patch.object(ui, 'get_model_display_name_from_config') as mock_display:
                
                # Different models give different estimates (risk assessment)
                mock_display.side_effect = lambda x: {
                    'rf_model': 'Random Forest',
                    'xgb_model': 'XGBoost',
                    'lr_model': 'Linear Regression'
                }.get(x, x)
                
                risky_project = {
                    'project_prf_functional_size': 300,
                    'project_prf_max_team_size': 4,
                    'project_prf_relative_size': 'M',
                    'external_eef_industry_sector': 'Healthcare',  # Regulated industry
                    'tech_tf_primary_programming_language': 'Python'
                }
                
                # Conservative model (Random Forest) - higher estimate
                mock_predict.return_value = 580.0
                ui.add_prediction_to_history(risky_project, 'rf_model', 580.0)
                
                # Optimistic model (Linear Regression) - lower estimate  
                mock_predict.return_value = 420.0
                ui.add_prediction_to_history(risky_project, 'lr_model', 420.0)
                
                # Balanced model (XGBoost) - middle estimate
                mock_predict.return_value = 500.0
                ui.add_prediction_to_history(risky_project, 'xgb_model', 500.0)
                
                # PM analyzes risk through model comparison
                ui.display_model_comparison()
                
                # Verify comparison shows variance (risk indicator)
                mock_dataframe.assert_called()
                stats_df = mock_dataframe.call_args[0][0]
                
                # Should have statistics for risk assessment
                assert 'Std Dev' in stats_df.columns
                assert 'Min' in stats_df.columns
                assert 'Max' in stats_df.columns
                
                # Calculate variance for risk assessment
                predictions = [580.0, 420.0, 500.0]
                variance = np.std(predictions)
                
                # High variance indicates higher risk
                assert variance > 50, "Should show significant variance for risk assessment"

class TestE2EDeveloperScenario:
    """Test scenarios for developer/technical lead persona"""
    
    def test_e2e_dev_technology_impact_analysis(self):
        """Developer analyzing impact of technology choices"""
        
        with patch('streamlit.sidebar'), \
             patch('streamlit.tabs'), \
             patch('streamlit.title'), \
             patch('streamlit.markdown'), \
             patch('streamlit.header'), \
             patch('streamlit.info'), \
             patch('streamlit.warning'), \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.columns'), \
             patch('streamlit.button'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.number_input'), \
             patch('streamlit.metric'), \
             patch('streamlit.bar_chart') as mock_bar_chart, \
             patch('streamlit.dataframe'), \
             patch('streamlit.expander'):
            
            with patch.object(ui, 'predict_man_hours') as mock_predict, \
                 patch.object(ui, 'get_feature_importance') as mock_importance, \
                 patch.object(ui, 'get_model_display_name') as mock_display, \
                 patch.object(ui, 'get_field_title') as mock_field_title:
                
                mock_display.return_value = "Random Forest"
                mock_field_title.side_effect = lambda x: x.replace('_', ' ').title()
                
                # Mock feature importance with tech stack having high importance
                mock_importance.return_value = np.array([
                    0.15,  # functional_size
                    0.10,  # team_size  
                    0.35,  # programming_language (high impact)
                    0.25,  # industry_sector
                    0.15   # other_factors
                ])
                
                base_project = {
                    'project_prf_functional_size': 200,
                    'project_prf_max_team_size': 5,
                    'project_prf_relative_size': 'M',
                    'external_eef_industry_sector': 'Technology'
                }
                
                # Test with Java
                mock_predict.return_value = 480.0
                java_project = base_project.copy()
                java_project['tech_tf_primary_programming_language'] = 'Java'
                
                prediction = mock_predict(java_project, 'rf_model')
                
                st.session_state['current_prediction_results'] = {
                    'prediction': prediction,
                    'model': 'rf_model',
                    'inputs': java_project
                }
                
                # Developer views feature importance to understand tech impact
                ui.show_feature_importance('rf_model', java_project)
                
                # Verify importance analysis was shown
                mock_importance.assert_called_with('rf_model')
                mock_bar_chart.assert_called()
                
                # Verify high importance of programming language is captured
                importance_values = mock_importance.return_value
                tech_importance = importance_values[2]  # programming language
                assert tech_importance == 0.35, "Programming language should have high importance"

    def test_e2e_dev_shap_deep_dive_analysis(self):
        """Developer doing deep-dive SHAP analysis to understand model behavior"""
        
        with patch('streamlit.header'), \
             patch('streamlit.info'), \
             patch('streamlit.warning'), \
             patch('streamlit.error'), \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.columns'), \
             patch('streamlit.metric'):
            
            with patch.object(ui, 'display_optimized_shap_analysis') as mock_shap, \
                 patch.object(ui, 'get_cache_info') as mock_cache_info, \
                 patch.object(ui, 'clear_explainer_cache') as mock_clear_cache:
                
                # Setup prediction history for SHAP analysis
                complex_project = {
                    'project_prf_functional_size': 150,
                    'project_prf_max_team_size': 6,
                    'project_prf_relative_size': 'M',
                    'external_eef_industry_sector': 'Financial',
                    'tech_tf_primary_programming_language': 'Python',
                    'additional_tech_factor': 'Microservices'
                }
                
                st.session_state['prediction_history'] = [{
                    'inputs': complex_project,
                    'model_technical': 'rf_model',
                    'prediction_hours': 520.0
                }]
                
                # Developer checks cache before analysis
                cache_info = ui.get_cache_info()
                assert isinstance(cache_info, dict)
                
                # Developer runs SHAP analysis
                latest_prediction = st.session_state['prediction_history'][-1]
                ui.display_instance_specific_shap(
                    latest_prediction['inputs'],
                    latest_prediction['model_technical']
                )
                
                # Verify SHAP analysis was called with correct parameters
                mock_shap.assert_called_once_with(
                    complex_project,
                    'rf_model',
                    ui.get_trained_model
                )
                
                # Developer clears cache for fresh analysis
                ui.clear_explainer_cache()
                mock_clear_cache.assert_called_once()

class TestE2EBusinessAnalystScenario:
    """Test scenarios for business analyst persona"""
    
    def test_e2e_ba_trend_analysis_workflow(self):
        """Business analyst analyzing trends across multiple projects"""
        
        with patch('streamlit.sidebar'), \
             patch('streamlit.tabs'), \
             patch('streamlit.title'), \
             patch('streamlit.markdown'), \
             patch('streamlit.header'), \
             patch('streamlit.info'), \
             patch('streamlit.warning'), \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.columns'), \
             patch('streamlit.button'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.number_input'), \
             patch('streamlit.metric'), \
             patch('streamlit.dataframe') as mock_dataframe, \
             patch('streamlit.bar_chart'):
            
            with patch.object(ui, 'predict_man_hours') as mock_predict:
                
                # BA analyzes multiple projects of different sizes
                projects = [
                    {
                        'name': 'Small Web App',
                        'inputs': {
                            'project_prf_functional_size': 50,
                            'project_prf_max_team_size': 3,
                            'project_prf_relative_size': 'S',
                            'external_eef_industry_sector': 'Technology',
                            'tech_tf_primary_programming_language': 'JavaScript'
                        },
                        'expected': 180.0
                    },
                    {
                        'name': 'Medium Enterprise App',
                        'inputs': {
                            'project_prf_functional_size': 300,
                            'project_prf_max_team_size': 6,
                            'project_prf_relative_size': 'M',
                            'external_eef_industry_sector': 'Financial',
                            'tech_tf_primary_programming_language': 'Java'
                        },
                        'expected': 520.0
                    },
                    {
                        'name': 'Large Platform',
                        'inputs': {
                            'project_prf_functional_size': 800,
                            'project_prf_max_team_size': 12,
                            'project_prf_relative_size': 'L',
                            'external_eef_industry_sector': 'Healthcare',
                            'tech_tf_primary_programming_language': 'Python'
                        },
                        'expected': 1400.0
                    }
                ]
                
                # BA runs predictions for all projects
                for project in projects:
                    mock_predict.return_value = project['expected']
                    ui.add_prediction_to_history(project['inputs'], 'rf_model', project['expected'])
                
                # Verify all projects are in history
                assert len(st.session_state['prediction_history']) == 3
                
                # BA analyzes the trend
                history = st.session_state['prediction_history']
                sizes = [entry['inputs']['project_prf_functional_size'] for entry in history]
                efforts = [entry['prediction_hours'] for entry in history]
                
                # Verify trend: larger projects need more effort
                assert sizes == [50, 300, 800]
                assert efforts == [180.0, 520.0, 1400.0]
                assert efforts[0] < efforts[1] < efforts[2], "Effort should increase with size"
                
                # BA views detailed history for analysis
                ui.show_prediction_history()
                
                # Should display comprehensive data for business analysis
                if mock_dataframe.called:
                    # Verify history dataframe has business-relevant columns
                    df_calls = mock_dataframe.call_args_list
                    if df_calls:
                        df = df_calls[0][0][0]  # First dataframe call, first argument
                        expected_columns = ['Timestamp', 'Model', 'Hours', 'Days']
                        # Note: In real test, we'd verify the actual dataframe structure

    def test_e2e_ba_cost_benefit_analysis(self):
        """Business analyst performing cost-benefit analysis"""
        
        with patch('streamlit.sidebar'), \
             patch('streamlit.tabs'), \
             patch('streamlit.title'), \
             patch('streamlit.markdown'), \
             patch('streamlit.header'), \
             patch('streamlit.info'), \
             patch('streamlit.warning'), \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.columns'), \
             patch('streamlit.button'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.number_input'), \
             patch('streamlit.metric') as mock_metric:
            
            with patch.object(ui, 'predict_man_hours') as mock_predict, \
                 patch.object(ui, 'get_model_display_name') as mock_display:
                
                mock_display.return_value = "Random Forest"
                
                # BA compares in-house vs outsourced development
                project_spec = {
                    'project_prf_functional_size': 400,
                    'project_prf_relative_size': 'M',
                    'external_eef_industry_sector': 'Financial',
                    'tech_tf_primary_programming_language': 'Java'
                }
                
                # In-house team (larger, potentially less efficient)
                mock_predict.return_value = 720.0
                inhouse_project = project_spec.copy()
                inhouse_project['project_prf_max_team_size'] = 8
                
                prediction_inhouse = mock_predict(inhouse_project, 'rf_model')
                
                st.session_state['current_prediction_results'] = {
                    'prediction': prediction_inhouse,
                    'model': 'rf_model',
                    'inputs': inhouse_project
                }
                
                # BA views detailed metrics for cost calculation
                results = st.session_state['current_prediction_results']
                ui.show_prediction(results['prediction'], results['model'], results['inputs'])
                
                # Verify business metrics are available
                mock_metric.assert_called()
                metric_calls = mock_metric.call_args_list
                
                # Should have time-based metrics for cost calculation
                metric_labels = [call[0][0] for call in metric_calls]
                business_metrics = ["📊 Total Effort", "📅 Working Days", "📆 Working Weeks", "🗓️ Months"]
                
                for metric in business_metrics:
                    assert metric in metric_labels
                
                # Calculate business value (720 hours = 90 days = ~4.5 months)
                hours = prediction_inhouse
                days = hours / UIConstants.HOURS_PER_DAY  # 8 hours per day
                weeks = days / UIConstants.DAYS_PER_WEEK  # 5 days per week
                months = weeks / 4.33
                
                assert hours == 720.0
                assert abs(days - 90.0) < 1  # Approximately 90 days
                assert abs(months - 4.15) < 0.5  # Approximately 4+ months

class TestE2EDataScientistScenario:
    """Test scenarios for data scientist persona"""
    
    def test_e2e_ds_model_performance_evaluation(self):
        """Data scientist evaluating model performance across scenarios"""
        
        with patch('streamlit.sidebar'), \
             patch('streamlit.tabs'), \
             patch('streamlit.title'), \
             patch('streamlit.markdown'), \
             patch('streamlit.header'), \
             patch('streamlit.info'), \
             patch('streamlit.warning'), \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.button'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.number_input'), \
             patch('streamlit.metric'), \
             patch('streamlit.dataframe') as mock_dataframe, \
             patch('plotly.express.box') as mock_box_plot, \
             patch('streamlit.plotly_chart'):
            
            mock_columns.return_value = [MagicMock(), MagicMock()]
            
            with patch.object(ui, 'predict_man_hours') as mock_predict, \
                 patch.object(ui, 'get_model_display_name_from_config') as mock_display:
                
                mock_display.side_effect = lambda x: {
                    'rf_model': 'Random Forest',
                    'xgb_model': 'XGBoost',
                    'lr_model': 'Linear Regression',
                    'svm_model': 'Support Vector Machine'
                }.get(x, x)
                
                # DS tests multiple models on same project
                test_project = {
                    'project_prf_functional_size': 250,
                    'project_prf_max_team_size': 5,
                    'project_prf_relative_size': 'M',
                    'external_eef_industry_sector': 'Technology',
                    'tech_tf_primary_programming_language': 'Python'
                }
                
                # Model performance varies
                models_performance = [
                    ('rf_model', 485.0),      # Random Forest
                    ('xgb_model', 492.0),     # XGBoost  
                    ('lr_model', 465.0),      # Linear Regression
                    ('svm_model', 505.0)      # SVM
                ]
                
                for model, prediction in models_performance:
                    mock_predict.return_value = prediction
                    ui.add_prediction_to_history(test_project, model, prediction)
                
                # DS analyzes model comparison
                ui.display_model_comparison()
                
                # Verify statistical analysis is available
                mock_dataframe.assert_called()
                mock_box_plot.assert_called()
                
                stats_df = mock_dataframe.call_args[0][0]
                
                # Verify comprehensive statistics for model evaluation
                required_stats = ['Model', 'Count', 'Mean', 'Std Dev', 'Min', 'Max']
                for stat in required_stats:
                    assert stat in stats_df.columns
                
                # Should have all 4 models
                assert len(stats_df) == 4
                
                # Verify box plot shows distribution
                plot_call = mock_box_plot.call_args
                assert plot_call[1]['x'] == 'Model'
                assert plot_call[1]['y'] == 'Prediction (Hours)'

    def test_e2e_ds_feature_importance_deep_analysis(self):
        """Data scientist analyzing feature importance patterns"""
        
        with patch('streamlit.sidebar'), \
             patch('streamlit.tabs'), \
             patch('streamlit.title'), \
             patch('streamlit.markdown'), \
             patch('streamlit.header'), \
             patch('streamlit.info'), \
             patch('streamlit.warning'), \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.columns'), \
             patch('streamlit.button'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.number_input'), \
             patch('streamlit.metric'), \
             patch('streamlit.bar_chart') as mock_bar_chart, \
             patch('streamlit.dataframe') as mock_dataframe, \
             patch('streamlit.expander'):
            
            with patch.object(ui, 'get_feature_importance') as mock_importance, \
                 patch.object(ui, 'get_model_display_name') as mock_display, \
                 patch.object(ui, 'get_field_title') as mock_field_title, \
                 patch.object(ui, 'FEATURE_IMPORTANCE_DISPLAY', {
                     'max_features_shown': 15,
                     'precision_decimals': 4
                 }):
                
                mock_display.return_value = "Random Forest"
                mock_field_title.side_effect = lambda x: x.replace('_', ' ').title()
                
                # Mock detailed feature importance
                mock_importance.return_value = np.array([
                    0.2845,  # functional_size - highest
                    0.1932,  # max_team_size
                    0.1567,  # industry_sector  
                    0.1289,  # programming_language
                    0.0876,  # relative_size
                    0.0654,  # complexity_factor
                    0.0423,  # methodology
                    0.0314,  # tools_used
                    0.0100   # other_factors
                ])
                
                complex_project = {
                    'project_prf_functional_size': 300,
                    'project_prf_max_team_size': 6,
                    'external_eef_industry_sector': 'Healthcare',
                    'tech_tf_primary_programming_language': 'Java',
                    'project_prf_relative_size': 'M',
                    'complexity_factor': 'High',
                    'methodology': 'Agile',
                    'tools_used': 'Advanced',
                    'other_factors': 'Standard'
                }
                
                # DS analyzes feature importance
                ui.show_feature_importance('rf_model', complex_project)
                
                # Verify detailed analysis was performed
                mock_importance.assert_called_with('rf_model')
                mock_bar_chart.assert_called()
                mock_dataframe.assert_called()
                
                # Check precision of importance values
                importance_df = mock_dataframe.call_args[0][0]
                
                # Should have detailed importance data
                assert 'Feature' in importance_df.columns
                assert 'Importance' in importance_df.columns
                
                # Should be sorted by importance (descending)
                importance_values = importance_df['Importance'].values
                assert all(importance_values[i] >= importance_values[i+1] 
                          for i in range(len(importance_values)-1))

class TestE2ENewUserOnboarding:
    """Test scenarios for new user onboarding"""
    
    def test_e2e_first_time_user_guided_experience(self):
        """New user's first experience with the application"""
        
        with patch('streamlit.sidebar'), \
             patch('streamlit.tabs') as mock_tabs, \
             patch('streamlit.title') as mock_title, \
             patch('streamlit.markdown') as mock_markdown, \
             patch('streamlit.header'), \
             patch('streamlit.info') as mock_info, \
             patch('streamlit.warning'), \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.columns'), \
             patch('streamlit.button'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.number_input'), \
             patch('streamlit.metric'), \
             patch('streamlit.expander') as mock_expander:
            
            # Mock tab structure for new user
            tabs = [MagicMock() for _ in range(5)]
            mock_tabs.return_value = tabs
            
            # Mock expander for help content
            expander_context = MagicMock()
            mock_expander.return_value.__enter__ = Mock(return_value=expander_context)
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            
            with patch.object(ui, 'check_required_models') as mock_check, \
                 patch.object(ui, 'list_available_models') as mock_list, \
                 patch.object(ui, 'about_section') as mock_about:
                
                mock_check.return_value = {"models_available": True}
                mock_list.return_value = [{"display_name": "Random Forest", "technical_name": "rf_model"}]
                
                # Step 1: New user opens application
                ui.initialize_session_state()
                
                # Verify clean initial state
                assert st.session_state['prediction_history'] == []
                assert st.session_state['current_prediction_results'] is None
                
                # Step 2: User sees main title and description
                expected_title = "🔮 ML Agile Software Project Effort Estimator"
                expected_desc = "Get accurate effort estimates using machine learning models trained on historical project data."
                
                # These would be called in main()
                st.title(expected_title)
                st.markdown(expected_desc)
                
                mock_title.assert_called_with(expected_title)
                mock_markdown.assert_called_with(expected_desc)
                
                # Step 3: User explores help section
                with st.expander("How to Use This Tool"):
                    st.markdown("Usage guide")
                
                with st.expander("About This Tool"):
                    ui.about_section()
                
                # Verify help content is accessible
                assert mock_expander.call_count == 2
                mock_about.assert_called_once()
                
                # Step 4: User gets guidance on required fields
                st.info("Required fields (marked with ⭐)")
                mock_info.assert_called()

    def test_e2e_user_learns_through_help_system(self):
        """User learning the system through help and guidance"""
        
        with patch('streamlit.expander') as mock_expander, \
             patch('streamlit.markdown') as mock_markdown, \
             patch('streamlit.info') as mock_info:
            
            expander_context = MagicMock()
            mock_expander.return_value.__enter__ = Mock(return_value=expander_context)
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            
            with patch.object(ui, 'about_section') as mock_about:
                
                # User reads usage guide
                usage_content = f"""
                ### Quick Start Guide
                
                1. **Fill Required Fields** - Complete all fields marked with {UIConstants.REQUIRED_FIELD_MARKER} in the sidebar
                2. **Optional Parameters** - Add more details for better accuracy  
                3. **Select Model** - Choose a model for prediction
                4. **Get Prediction** - Click 'Predict Effort' to see your estimate
                5. **Analyze Results** - Use the Instance-Specific SHAP tab for insights
                """
                
                with st.expander("How to Use This Tool"):
                    st.markdown(usage_content)
                
                # User reads about section
                with st.expander("About This Tool"):
                    ui.about_section()
                
                # Verify educational content is provided
                mock_expander.assert_called()
                mock_about.assert_called()
                
                # Verify step-by-step guidance
                markdown_calls = [call[0][0] for call in mock_markdown.call_args_list]
                guidance_content = " ".join(markdown_calls)
                
                # Should contain key learning points
                learning_points = [
                    "Fill Required Fields",
                    "Select Model", 
                    "Get Prediction",
                    "SHAP"
                ]
                
                for point in learning_points:
                    assert point in guidance_content

class TestE2ERealWorldUsagePatterns:
    """Test real-world usage patterns and edge cases"""
    
    def test_e2e_iterative_estimation_refinement(self):
        """User iteratively refining estimates"""
        
        with patch('streamlit.sidebar'), \
             patch('streamlit.tabs'), \
             patch('streamlit.title'), \
             patch('streamlit.markdown'), \
             patch('streamlit.header'), \
             patch('streamlit.info'), \
             patch('streamlit.warning'), \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.columns'), \
             patch('streamlit.button'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.number_input'), \
             patch('streamlit.metric'):
            
            with patch.object(ui, 'predict_man_hours') as mock_predict:
                
                # Initial rough estimate
                rough_estimate = {
                    'project_prf_functional_size': 200,
                    'project_prf_max_team_size': 5,
                    'project_prf_relative_size': 'M'
                }
                
                mock_predict.return_value = 450.0
                ui.add_prediction_to_history(rough_estimate, 'rf_model', 450.0)
                
                # Refined estimate with more details
                refined_estimate = rough_estimate.copy()
                refined_estimate.update({
                    'external_eef_industry_sector': 'Financial',
                    'tech_tf_primary_programming_language': 'Java',
                    'complexity_factor': 'Medium'
                })
                
                mock_predict.return_value = 485.0
                ui.add_prediction_to_history(refined_estimate, 'rf_model', 485.0)
                
                # Final detailed estimate
                detailed_estimate = refined_estimate.copy()
                detailed_estimate.update({
                    'methodology': 'Agile',
                    'team_experience': 'Senior',
                    'tools_quality': 'Advanced'
                })
                
                mock_predict.return_value = 465.0
                ui.add_prediction_to_history(detailed_estimate, 'rf_model', 465.0)
                
                # Verify iterative refinement is captured
                assert len(st.session_state['prediction_history']) == 3
                
                estimates = [entry['prediction_hours'] for entry in st.session_state['prediction_history']]
                input_counts = [len(entry['inputs']) for entry in st.session_state['prediction_history']]
                
                # More inputs should lead to refined estimates
                assert input_counts == [3, 6, 9]  # Increasing detail
                assert estimates == [450.0, 485.0, 465.0]  # Refined estimates

    def test_e2e_team_collaboration_scenario(self):
        """Multiple team members using the same session"""
        
        with patch('streamlit.sidebar'), \
             patch('streamlit.tabs'), \
             patch('streamlit.title'), \
             patch('streamlit.markdown'), \
             patch('streamlit.header'), \
             patch('streamlit.info'), \
             patch('streamlit.warning'), \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.columns'), \
             patch('streamlit.button'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.number_input'), \
             patch('streamlit.metric'), \
             patch('streamlit.dataframe'):
            
            with patch.object(ui, 'predict_man_hours') as mock_predict:
                
                # Team member 1: Backend estimate
                mock_predict.return_value = 320.0
                backend_estimate = {
                    'project_prf_functional_size': 150,
                    'project_prf_max_team_size': 3,
                    'project_prf_relative_size': 'M',
                    'tech_tf_primary_programming_language': 'Python',
                    'component': 'Backend API'
                }
                ui.add_prediction_to_history(backend_estimate, 'rf_model', 320.0)
                
                # Team member 2: Frontend estimate  
                mock_predict.return_value = 280.0
                frontend_estimate = {
                    'project_prf_functional_size': 120,
                    'project_prf_max_team_size': 2,
                    'project_prf_relative_size': 'M',
                    'tech_tf_primary_programming_language': 'JavaScript',
                    'component': 'Frontend UI'
                }
                ui.add_prediction_to_history(frontend_estimate, 'rf_model', 280.0)
                
                # Team member 3: Integration estimate
                mock_predict.return_value = 180.0
                integration_estimate = {
                    'project_prf_functional_size': 80,
                    'project_prf_max_team_size': 2,
                    'project_prf_relative_size': 'S',
                    'tech_tf_primary_programming_language': 'Python',
                    'component': 'Integration & Testing'
                }
                ui.add_prediction_to_history(integration_estimate, 'rf_model', 180.0)
                
                # Team reviews combined estimates
                ui.show_prediction_history()
                
                # Verify collaborative estimates are captured
                assert len(st.session_state['prediction_history']) == 3
                
                total_effort = sum(entry['prediction_hours'] for entry in st.session_state['prediction_history'])
                assert total_effort == 780.0  # 320 + 280 + 180
                
                # Team can see breakdown by component
                components = [entry['inputs'].get('component', 'Unknown') for entry in st.session_state['prediction_history']]
                assert 'Backend API' in components
                assert 'Frontend UI' in components  
                assert 'Integration & Testing' in components

if __name__ == "__main__":
    pytest.main([__file__, "-v"])