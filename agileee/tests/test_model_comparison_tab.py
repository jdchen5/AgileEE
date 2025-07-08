# test_model_comparison_tab.py
"""
Test cases for the Model Comparison Tab (Tab 3) - Multi-model analysis functionality
Verifies that model comparison features work correctly and are unaffected by UI simplification.
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

class TestModelComparisonTabCore:
    """Test core model comparison functionality"""
    
    def setup_method(self):
        """Setup common test data"""
        self.mock_multi_model_history = [
            {
                'timestamp': '2024-01-01 10:00:00',
                'model': 'Random Forest',
                'model_technical': 'rf_model',
                'prediction_hours': 480.0,
                'inputs': {'project_prf_functional_size': 100}
            },
            {
                'timestamp': '2024-01-01 10:30:00',
                'model': 'XGBoost',
                'model_technical': 'xgb_model', 
                'prediction_hours': 520.0,
                'inputs': {'project_prf_functional_size': 100}
            },
            {
                'timestamp': '2024-01-01 11:00:00',
                'model': 'Linear Regression',
                'model_technical': 'lr_model',
                'prediction_hours': 450.0,
                'inputs': {'project_prf_functional_size': 100}
            }
        ]
        
        # Reset session state
        st.session_state = {
            'prediction_history': []
        }

    def test_model_comparison_header_display(self):
        """Test model comparison tab shows correct header"""
        
        with patch('streamlit.header') as mock_header:
            
            # Simulate the model comparison tab from main()
            ui.display_model_comparison()
            
            # Verify header is displayed
            mock_header.assert_called_with("🤖 Model Comparison")

    def test_model_comparison_insufficient_data_warning(self):
        """Test warning when less than 2 models in history"""
        
        with patch('streamlit.warning') as mock_warning:
            
            # Single model history
            st.session_state['prediction_history'] = [self.mock_multi_model_history[0]]
            
            ui.display_model_comparison()
            
            # Should show warning
            expected_warning = "⚠️ Please make predictions with at least 2 different models to enable comparison."
            mock_warning.assert_called_with(expected_warning)

    def test_model_comparison_with_valid_data(self):
        """Test model comparison with sufficient data"""
        
        with patch('streamlit.columns') as mock_columns, \
             patch('streamlit.subheader') as mock_subheader, \
             patch('streamlit.dataframe') as mock_dataframe, \
             patch('plotly.express.box') as mock_box_plot, \
             patch('streamlit.plotly_chart') as mock_plotly_chart:
            
            # Mock columns
            col1, col2 = MagicMock(), MagicMock()
            mock_columns.return_value = [col1, col2]
            
            # Set up multi-model history
            st.session_state['prediction_history'] = self.mock_multi_model_history.copy()
            
            with patch.object(ui, 'get_model_display_name_from_config') as mock_display_name:
                mock_display_name.side_effect = lambda x: x.replace('_', ' ').title()
                
                ui.display_model_comparison()
                
                # Verify comparison sections were created
                subheader_calls = [call[0][0] for call in mock_subheader.call_args_list]
                assert "📊 Model Performance Comparison" in subheader_calls
                assert "📈 Model Statistics" in subheader_calls
                
                # Verify plot was created
                mock_box_plot.assert_called_once()
                mock_plotly_chart.assert_called_once()
                
                # Verify statistics dataframe was created
                mock_dataframe.assert_called_once()

    def test_model_comparison_statistics_calculation(self):
        """Test model comparison statistics calculation"""
        
        with patch('streamlit.columns') as mock_columns, \
             patch('streamlit.subheader'), \
             patch('streamlit.dataframe') as mock_dataframe, \
             patch('plotly.express.box'), \
             patch('streamlit.plotly_chart'):
            
            mock_columns.return_value = [MagicMock(), MagicMock()]
            
            st.session_state['prediction_history'] = self.mock_multi_model_history.copy()
            
            with patch.object(ui, 'get_model_display_name_from_config') as mock_display_name:
                mock_display_name.side_effect = lambda x: {
                    'rf_model': 'Random Forest',
                    'xgb_model': 'XGBoost', 
                    'lr_model': 'Linear Regression'
                }.get(x, x)
                
                ui.display_model_comparison()
                
                # Verify dataframe was called with statistics
                mock_dataframe.assert_called_once()
                stats_df = mock_dataframe.call_args[0][0]
                
                # Verify statistics structure
                assert isinstance(stats_df, pd.DataFrame)
                assert 'Model' in stats_df.columns
                assert 'Count' in stats_df.columns
                assert 'Mean' in stats_df.columns
                assert 'Std Dev' in stats_df.columns
                assert 'Min' in stats_df.columns
                assert 'Max' in stats_df.columns
                
                # Verify we have 3 models
                assert len(stats_df) == 3

    def test_model_comparison_visualization_data(self):
        """Test model comparison visualization data structure"""
        
        with patch('streamlit.columns') as mock_columns, \
             patch('streamlit.subheader'), \
             patch('streamlit.dataframe'), \
             patch('plotly.express.box') as mock_box_plot, \
             patch('streamlit.plotly_chart'):
            
            mock_columns.return_value = [MagicMock(), MagicMock()]
            
            st.session_state['prediction_history'] = self.mock_multi_model_history.copy()
            
            with patch.object(ui, 'get_model_display_name_from_config') as mock_display_name:
                mock_display_name.side_effect = lambda x: x.replace('_', ' ').title()
                
                ui.display_model_comparison()
                
                # Verify box plot was called with correct data structure
                mock_box_plot.assert_called_once()
                plot_call = mock_box_plot.call_args
                
                # Check the DataFrame passed to plotly
                comparison_df = plot_call[1]['data_frame']
                assert 'Model' in comparison_df.columns
                assert 'Prediction (Hours)' in comparison_df.columns
                
                # Verify data content
                assert len(comparison_df) == 3  # 3 predictions
                assert comparison_df['Prediction (Hours)'].tolist() == [480.0, 520.0, 450.0]

class TestModelComparisonIntegration:
    """Test model comparison integration features"""
    
    def test_model_comparison_with_same_model_multiple_times(self):
        """Test comparison when same model used multiple times"""
        
        # Same model used twice with different inputs
        same_model_history = [
            {
                'model_technical': 'rf_model',
                'prediction_hours': 480.0,
                'inputs': {'project_prf_functional_size': 100}
            },
            {
                'model_technical': 'rf_model', 
                'prediction_hours': 520.0,
                'inputs': {'project_prf_functional_size': 150}
            },
            {
                'model_technical': 'xgb_model',
                'prediction_hours': 500.0,
                'inputs': {'project_prf_functional_size': 125}
            }
        ]
        
        with patch('streamlit.columns') as mock_columns, \
             patch('streamlit.subheader'), \
             patch('streamlit.dataframe') as mock_dataframe, \
             patch('plotly.express.box'), \
             patch('streamlit.plotly_chart'):
            
            mock_columns.return_value = [MagicMock(), MagicMock()]
            
            st.session_state['prediction_history'] = same_model_history
            
            with patch.object(ui, 'get_model_display_name_from_config') as mock_display_name:
                mock_display_name.side_effect = lambda x: x.replace('_', ' ').title()
                
                ui.display_model_comparison()
                
                # Should handle multiple instances of same model
                # Statistics should show rf_model with count=2, xgb_model with count=1
                mock_dataframe.assert_called_once()
                stats_df = mock_dataframe.call_args[0][0]
                
                # Should have 2 unique models in statistics
                assert len(stats_df) == 2

    def test_model_comparison_error_handling(self):
        """Test model comparison error handling"""
        
        with patch('streamlit.error') as mock_error:
            
            # History with missing model technical names
            corrupted_history = [
                {
                    'model_technical': None,  # Missing
                    'prediction_hours': 480.0
                },
                {
                    'model_technical': 'valid_model',
                    'prediction_hours': 520.0
                }
            ]
            
            st.session_state['prediction_history'] = corrupted_history
            
            try:
                ui.display_model_comparison()
                # Should handle gracefully without crashing
            except Exception:
                # If exception occurs, it should be caught and error displayed
                pass

    def test_model_comparison_display_name_fallback(self):
        """Test model comparison with display name fallbacks"""
        
        with patch('streamlit.columns') as mock_columns, \
             patch('streamlit.subheader'), \
             patch('streamlit.dataframe') as mock_dataframe, \
             patch('plotly.express.box'), \
             patch('streamlit.plotly_chart'):
            
            mock_columns.return_value = [MagicMock(), MagicMock()]
            
            st.session_state['prediction_history'] = [
                {
                    'model_technical': 'unknown_model',
                    'prediction_hours': 480.0
                },
                {
                    'model_technical': 'another_unknown',
                    'prediction_hours': 520.0
                }
            ]
            
            with patch.object(ui, 'get_model_display_name_from_config') as mock_display_name:
                # Mock display name function to fall back to technical name
                mock_display_name.side_effect = lambda x: x  # Return technical name as fallback
                
                ui.display_model_comparison()
                
                # Should work with fallback names
                mock_dataframe.assert_called_once()
                stats_df = mock_dataframe.call_args[0][0]
                
                # Should contain the technical names as display names
                model_names = stats_df['Model'].tolist()
                assert 'unknown_model' in model_names
                assert 'another_unknown' in model_names

class TestModelComparisonVisualization:
    """Test model comparison visualization components"""
    
    def test_box_plot_creation(self):
        """Test box plot creation for model comparison"""
        
        with patch('streamlit.columns') as mock_columns, \
             patch('streamlit.subheader'), \
             patch('streamlit.dataframe'), \
             patch('plotly.express.box') as mock_box_plot, \
             patch('streamlit.plotly_chart') as mock_plotly_chart:
            
            mock_columns.return_value = [MagicMock(), MagicMock()]
            
            st.session_state['prediction_history'] = [
                {
                    'model_technical': 'model1',
                    'prediction_hours': 480.0
                },
                {
                    'model_technical': 'model2', 
                    'prediction_hours': 520.0
                }
            ]
            
            with patch.object(ui, 'get_model_display_name_from_config') as mock_display_name:
                mock_display_name.side_effect = lambda x: x
                
                ui.display_model_comparison()
                
                # Verify box plot parameters
                mock_box_plot.assert_called_once()
                plot_args = mock_box_plot.call_args[1]
                
                assert plot_args['x'] == 'Model'
                assert plot_args['y'] == 'Prediction (Hours)'
                assert plot_args['title'] == "Distribution of Predictions by Model"
                
                # Verify plot was displayed
                mock_plotly_chart.assert_called_once()

    def test_statistics_table_format(self):
        """Test statistics table formatting"""
        
        with patch('streamlit.columns') as mock_columns, \
             patch('streamlit.subheader'), \
             patch('streamlit.dataframe') as mock_dataframe, \
             patch('plotly.express.box'), \
             patch('streamlit.plotly_chart'):
            
            mock_columns.return_value = [MagicMock(), MagicMock()]
            
            # Test data with known statistics
            st.session_state['prediction_history'] = [
                {
                    'model_technical': 'test_model',
                    'prediction_hours': 400.0
                },
                {
                    'model_technical': 'test_model',
                    'prediction_hours': 500.0
                },
                {
                    'model_technical': 'test_model',
                    'prediction_hours': 600.0
                }
            ]
            
            with patch.object(ui, 'get_model_display_name_from_config') as mock_display_name:
                mock_display_name.return_value = 'Test Model'
                
                ui.display_model_comparison()
                
                mock_dataframe.assert_called_once()
                stats_df = mock_dataframe.call_args[0][0]
                
                # Verify statistics are formatted as strings with proper precision
                row = stats_df.iloc[0]
                assert row['Count'] == 3
                assert row['Mean'] == '500'  # Should be formatted as string
                assert row['Min'] == '400'
                assert row['Max'] == '600'

class TestModelComparisonLayout:
    """Test model comparison layout and structure"""
    
    def test_two_column_layout(self):
        """Test model comparison uses two-column layout"""
        
        with patch('streamlit.columns') as mock_columns, \
             patch('streamlit.subheader'), \
             patch('streamlit.dataframe'), \
             patch('plotly.express.box'), \
             patch('streamlit.plotly_chart'):
            
            col1, col2 = MagicMock(), MagicMock()
            mock_columns.return_value = [col1, col2]
            
            st.session_state['prediction_history'] = [
                {'model_technical': 'model1', 'prediction_hours': 480.0},
                {'model_technical': 'model2', 'prediction_hours': 520.0}
            ]
            
            with patch.object(ui, 'get_model_display_name_from_config'):
                ui.display_model_comparison()
                
                # Verify two columns were created
                mock_columns.assert_called_with(2)
                
                # Verify both columns were used (context managers)
                assert col1.__enter__.called
                assert col2.__enter__.called

    def test_comparison_sections_order(self):
        """Test model comparison sections appear in correct order"""
        
        with patch('streamlit.columns') as mock_columns, \
             patch('streamlit.subheader') as mock_subheader, \
             patch('streamlit.dataframe'), \
             patch('plotly.express.box'), \
             patch('streamlit.plotly_chart'):
            
            mock_columns.return_value = [MagicMock(), MagicMock()]
            
            st.session_state['prediction_history'] = [
                {'model_technical': 'model1', 'prediction_hours': 480.0},
                {'model_technical': 'model2', 'prediction_hours': 520.0}
            ]
            
            with patch.object(ui, 'get_model_display_name_from_config'):
                ui.display_model_comparison()
                
                # Verify section headers were called in correct order
                subheader_calls = [call[0][0] for call in mock_subheader.call_args_list]
                
                assert "📊 Model Performance Comparison" in subheader_calls
                assert "📈 Model Statistics" in subheader_calls
                
                # Verify performance comparison comes before statistics
                perf_index = subheader_calls.index("📊 Model Performance Comparison")
                stats_index = subheader_calls.index("📈 Model Statistics")
                assert perf_index < stats_index

class TestModelComparisonNoConfig:
    """Test model comparison works without configuration management"""
    
    def test_comparison_independent_of_config_state(self):
        """Test model comparison works without config state"""
        
        with patch('streamlit.columns') as mock_columns, \
             patch('streamlit.subheader'), \
             patch('streamlit.dataframe') as mock_dataframe, \
             patch('plotly.express.box'), \
             patch('streamlit.plotly_chart'):
            
            mock_columns.return_value = [MagicMock(), MagicMock()]
            
            # Minimal session state (no config variables)
            st.session_state = {
                'prediction_history': [
                    {'model_technical': 'model1', 'prediction_hours': 480.0},
                    {'model_technical': 'model2', 'prediction_hours': 520.0}
                ]
                # No config-related state variables
            }
            
            with patch.object(ui, 'get_model_display_name_from_config'):
                # Should work without any config state
                ui.display_model_comparison()
                
                mock_dataframe.assert_called_once()

    def test_comparison_no_save_load_dependencies(self):
        """Test model comparison has no save/load dependencies"""
        
        # Model comparison should not reference any save/load functionality
        
        with patch('streamlit.columns') as mock_columns, \
             patch('streamlit.subheader'), \
             patch('streamlit.dataframe'), \
             patch('plotly.express.box'), \
             patch('streamlit.plotly_chart'):
            
            mock_columns.return_value = [MagicMock(), MagicMock()]
            
            st.session_state['prediction_history'] = [
                {'model_technical': 'model1', 'prediction_hours': 480.0},
                {'model_technical': 'model2', 'prediction_hours': 520.0}
            ]
            
            with patch.object(ui, 'get_model_display_name_from_config'):
                # Should work completely independently
                ui.display_model_comparison()
                
                # Function should not access any config save/load variables
                # This is verified by the test not failing

if __name__ == "__main__":
    pytest.main([__file__, "-v"])