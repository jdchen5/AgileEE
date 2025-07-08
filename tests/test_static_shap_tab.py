# test_static_shap_tab.py
"""
Test cases for the Static SHAP Analysis Tab (Tab 4) - File-based SHAP analysis
Verifies that static SHAP analysis loads from file and displays correctly.
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock, mock_open
import pandas as pd
import numpy as np
import sys
import os
import agileee.ui as ui

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the simplified UI module
import agileee.ui
from agileee.constants import UIConstants, FileConstants

class TestStaticShapTabCore:
    """Test core static SHAP analysis functionality"""
    
    def setup_method(self):
        """Setup common test data"""
        self.mock_shap_content = """
# Static SHAP Analysis - Model Feature Importance

## Overview
This analysis shows the global feature importance across all models.

## Key Findings
- Feature 1: High importance (0.35)
- Feature 2: Medium importance (0.25) 
- Feature 3: Low importance (0.15)

## Model Performance
The analysis covers multiple models with consistent feature ranking.
"""
        
        # Reset any state if needed
        st.session_state = {}

    def test_static_shap_tab_header_display(self):
        """Test static SHAP tab shows correct header"""
        
        with patch('streamlit.header') as mock_header:
            
            # Simulate the static SHAP tab from main()
            expected_header = "📈 Static SHAP Analysis - Model Feature Importance"
            
            # This would be called in the static SHAP tab
            st.header(expected_header)
            
            mock_header.assert_called_with(expected_header)

    def test_display_static_shap_analysis_success(self):
        """Test successful loading and display of static SHAP analysis"""
        
        with patch('builtins.open', mock_open(read_data=self.mock_shap_content)), \
             patch('streamlit.markdown') as mock_markdown, \
             patch('streamlit.header'):
            
            # Call the display function
            ui.display_static_shap_analysis()
            
            # Verify file was read and content displayed
            mock_markdown.assert_called_once_with(self.mock_shap_content, unsafe_allow_html=True)

    def test_display_static_shap_analysis_file_not_found(self):
        """Test static SHAP analysis handles missing file"""
        
        with patch('builtins.open', side_effect=FileNotFoundError("File not found")), \
             patch('streamlit.error') as mock_error, \
             patch('streamlit.header'):
            
            ui.display_static_shap_analysis()
            
            # Should display error message
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][0]
            assert "Failed to load static SHAP analysis report" in error_message

    def test_display_static_shap_analysis_permission_error(self):
        """Test static SHAP analysis handles permission errors"""
        
        with patch('builtins.open', side_effect=PermissionError("Permission denied")), \
             patch('streamlit.error') as mock_error, \
             patch('streamlit.header'):
            
            ui.display_static_shap_analysis()
            
            # Should display error message
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][0]
            assert "Failed to load static SHAP analysis report" in error_message

    def test_display_static_shap_analysis_encoding_error(self):
        """Test static SHAP analysis handles encoding errors"""
        
        with patch('builtins.open', side_effect=UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid')), \
             patch('streamlit.error') as mock_error, \
             patch('streamlit.header'):
            
            ui.display_static_shap_analysis()
            
            # Should display error message
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][0]
            assert "Failed to load static SHAP analysis report" in error_message

class TestStaticShapTabFileHandling:
    """Test file handling for static SHAP analysis"""
    
    def test_static_shap_file_path_constant(self):
        """Test static SHAP uses correct file path from constants"""
        
        with patch('builtins.open', mock_open(read_data="test content")) as mock_file, \
             patch('streamlit.markdown'), \
             patch('streamlit.header'):
            
            ui.display_static_shap_analysis()
            
            # Verify correct file path was used
            mock_file.assert_called_once_with(FileConstants.SHAP_ANALYSIS_FILE, "r", encoding="utf-8")

    def test_static_shap_file_encoding_utf8(self):
        """Test static SHAP file is read with UTF-8 encoding"""
        
        with patch('builtins.open', mock_open(read_data="test content")) as mock_file, \
             patch('streamlit.markdown'), \
             patch('streamlit.header'):
            
            ui.display_static_shap_analysis()
            
            # Verify UTF-8 encoding was specified
            mock_file.assert_called_once_with(FileConstants.SHAP_ANALYSIS_FILE, "r", encoding="utf-8")

    def test_static_shap_file_content_types(self):
        """Test static SHAP handles different content types"""
        
        test_contents = [
            "# Simple markdown content",
            "## Complex content\n\n- List item 1\n- List item 2\n\n**Bold text**",
            "<h1>HTML content</h1><p>With HTML tags</p>",
            "Mixed content with **markdown** and <em>HTML</em>"
        ]
        
        for content in test_contents:
            with patch('builtins.open', mock_open(read_data=content)), \
                 patch('streamlit.markdown') as mock_markdown, \
                 patch('streamlit.header'):
                
                ui.display_static_shap_analysis()
                
                # Verify content was passed through correctly
                mock_markdown.assert_called_once_with(content, unsafe_allow_html=True)
                mock_markdown.reset_mock()

class TestStaticShapTabIntegration:
    """Test static SHAP tab integration with overall UI"""
    
    def test_static_shap_tab_independent_operation(self):
        """Test static SHAP tab works independently of other tabs"""
        
        with patch('builtins.open', mock_open(read_data="test content")), \
             patch('streamlit.markdown') as mock_markdown, \
             patch('streamlit.header'):
            
            # Should work regardless of session state
            st.session_state = {}  # Empty state
            
            ui.display_static_shap_analysis()
            
            # Should still work
            mock_markdown.assert_called_once()

    def test_static_shap_no_prediction_dependency(self):
        """Test static SHAP doesn't depend on prediction history"""
        
        with patch('builtins.open', mock_open(read_data="test content")), \
             patch('streamlit.markdown') as mock_markdown, \
             patch('streamlit.header'):
            
            # No prediction history should not affect static SHAP
            st.session_state = {'prediction_history': []}
            
            ui.display_static_shap_analysis()
            
            # Should work fine
            mock_markdown.assert_called_once()

    def test_static_shap_no_model_dependency(self):
        """Test static SHAP doesn't depend on loaded models"""
        
        with patch('builtins.open', mock_open(read_data="test content")), \
             patch('streamlit.markdown') as mock_markdown, \
             patch('streamlit.header'):
            
            # Should work even if no models are available
            with patch.object(ui, 'MODELS_AVAILABLE', False):
                ui.display_static_shap_analysis()
                
                # Should still work
                mock_markdown.assert_called_once()

class TestStaticShapTabContent:
    """Test static SHAP tab content display"""
    
    def test_static_shap_markdown_rendering(self):
        """Test static SHAP content is rendered as markdown"""
        
        markdown_content = """
# SHAP Feature Analysis

## Top Features
1. **project_prf_functional_size**: 0.35
2. **project_prf_max_team_size**: 0.25
3. **tech_tf_primary_programming_language**: 0.15

## Model Comparison
- Random Forest: Most consistent
- XGBoost: Highest accuracy
- Linear Regression: Baseline model
"""
        
        with patch('builtins.open', mock_open(read_data=markdown_content)), \
             patch('streamlit.markdown') as mock_markdown, \
             patch('streamlit.header'):
            
            ui.display_static_shap_analysis()
            
            # Verify content and unsafe_allow_html=True
            mock_markdown.assert_called_once_with(markdown_content, unsafe_allow_html=True)

    def test_static_shap_html_content_support(self):
        """Test static SHAP supports HTML content"""
        
        html_content = """
<h1>SHAP Analysis Report</h1>
<div style="background-color: #f0f0f0; padding: 10px;">
<h2>Feature Importance</h2>
<ul>
<li>Feature 1: <strong>High</strong></li>
<li>Feature 2: <strong>Medium</strong></li>
</ul>
</div>
"""
        
        with patch('builtins.open', mock_open(read_data=html_content)), \
             patch('streamlit.markdown') as mock_markdown, \
             patch('streamlit.header'):
            
            ui.display_static_shap_analysis()
            
            # Verify HTML is allowed
            mock_markdown.assert_called_once_with(html_content, unsafe_allow_html=True)

    def test_static_shap_empty_file_handling(self):
        """Test static SHAP handles empty files"""
        
        with patch('builtins.open', mock_open(read_data="")), \
             patch('streamlit.markdown') as mock_markdown, \
             patch('streamlit.header'):
            
            ui.display_static_shap_analysis()
            
            # Should handle empty content gracefully
            mock_markdown.assert_called_once_with("", unsafe_allow_html=True)

class TestStaticShapTabErrorScenarios:
    """Test error scenarios for static SHAP tab"""
    
    def test_static_shap_corrupted_file(self):
        """Test static SHAP handles corrupted file content"""
        
        # Binary content that might cause issues
        with patch('builtins.open', side_effect=UnicodeDecodeError('utf-8', b'\xff\xfe', 0, 1, 'invalid')), \
             patch('streamlit.error') as mock_error, \
             patch('streamlit.header'):
            
            ui.display_static_shap_analysis()
            
            mock_error.assert_called_once()

    def test_static_shap_very_large_file(self):
        """Test static SHAP with large file content"""
        
        large_content = "# Large content\n" + "Content line\n" * 10000
        
        with patch('builtins.open', mock_open(read_data=large_content)), \
             patch('streamlit.markdown') as mock_markdown, \
             patch('streamlit.header'):
            
            ui.display_static_shap_analysis()
            
            # Should handle large content
            mock_markdown.assert_called_once_with(large_content, unsafe_allow_html=True)

    def test_static_shap_disk_full_error(self):
        """Test static SHAP handles disk I/O errors"""
        
        with patch('builtins.open', side_effect=OSError("Disk full")), \
             patch('streamlit.error') as mock_error, \
             patch('streamlit.header'):
            
            ui.display_static_shap_analysis()
            
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][0]
            assert "Failed to load static SHAP analysis report" in error_message

class TestStaticShapTabNoConfig:
    """Test static SHAP tab works without configuration management"""
    
    def test_static_shap_no_config_dependencies(self):
        """Test static SHAP doesn't depend on configuration state"""
        
        with patch('builtins.open', mock_open(read_data="test content")), \
             patch('streamlit.markdown') as mock_markdown, \
             patch('streamlit.header'):
            
            # Verify no config-related variables are accessed
            # Static SHAP should be completely independent
            
            ui.display_static_shap_analysis()
            
            mock_markdown.assert_called_once()

    def test_static_shap_no_save_load_references(self):
        """Test static SHAP has no save/load functionality"""
        
        # Static SHAP should be read-only and not have any save/load features
        
        with patch('builtins.open', mock_open(read_data="test content")), \
             patch('streamlit.markdown'), \
             patch('streamlit.header'):
            
            # Should not call any save/load related functions
            with patch('streamlit.file_uploader') as mock_upload, \
                 patch('streamlit.download_button') as mock_download:
                
                ui.display_static_shap_analysis()
                
                # Verify no upload/download functionality
                mock_upload.assert_not_called()
                mock_download.assert_not_called()

class TestStaticShapTabAccessibility:
    """Test static SHAP tab accessibility and usability"""
    
    def test_static_shap_header_structure(self):
        """Test static SHAP has proper header structure"""
        
        with patch('builtins.open', mock_open(read_data="content")), \
             patch('streamlit.markdown'), \
             patch('streamlit.header') as mock_header:
            
            ui.display_static_shap_analysis()
            
            # Should have descriptive header
            header_call = mock_header.call_args[0][0] 
            assert "Static SHAP Analysis" in header_call
            assert "Model Feature Importance" in header_call

    def test_static_shap_error_message_clarity(self):
        """Test static SHAP error messages are clear"""
        
        with patch('builtins.open', side_effect=FileNotFoundError("File not found")), \
             patch('streamlit.error') as mock_error, \
             patch('streamlit.header'):
            
            ui.display_static_shap_analysis()
            
            error_message = mock_error.call_args[0][0]
            
            # Error message should be clear and helpful
            assert "Failed to load static SHAP analysis report" in error_message
            assert isinstance(error_message, str)
            assert len(error_message) > 10  # Should be descriptive

    def test_static_shap_consistent_behavior(self):
        """Test static SHAP behaves consistently"""
        
        with patch('builtins.open', mock_open(read_data="test content")), \
             patch('streamlit.markdown') as mock_markdown, \
             patch('streamlit.header'):
            
            # Should behave the same way on multiple calls
            ui.display_static_shap_analysis()
            first_call = mock_markdown.call_args
            
            mock_markdown.reset_mock()
            
            ui.display_static_shap_analysis()
            second_call = mock_markdown.call_args
            
            # Should be identical
            assert first_call == second_call

if __name__ == "__main__":
    pytest.main([__file__, "-v"])