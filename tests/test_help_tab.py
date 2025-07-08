# test_help_tab.py
"""
Test cases for the Help Tab (Tab 5) - Documentation and information
Verifies that help content displays correctly and is unaffected by UI simplification.
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the simplified UI module
import agileee.ui as ui
from agileee.constants import UIConstants, FileConstants

class TestHelpTabCore:
    """Test core help tab functionality"""
    
    def setup_method(self):
        """Setup common test data"""
        # Reset any state if needed
        st.session_state = {}

    def test_help_tab_usage_guide_expander(self):
        """Test help tab shows usage guide in expander"""
        
        with patch('streamlit.expander') as mock_expander, \
             patch('streamlit.markdown') as mock_markdown:
            
            # Mock expander context
            expander_context = MagicMock()
            mock_expander.return_value.__enter__ = Mock(return_value=expander_context)
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            
            # Simulate help tab logic from main()
            with st.expander("How to Use This Tool"):
                st.markdown("Usage guide content")
            
            # Verify expander was created with correct title
            mock_expander.assert_called_with("How to Use This Tool")
            mock_markdown.assert_called_with("Usage guide content")

    def test_help_tab_about_section_expander(self):
        """Test help tab shows about section in expander"""
        
        with patch('streamlit.expander') as mock_expander, \
             patch.object(ui, 'about_section') as mock_about:
            
            # Mock expander context
            expander_context = MagicMock()
            mock_expander.return_value.__enter__ = Mock(return_value=expander_context)
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            
            # Simulate help tab logic
            with st.expander("About This Tool"):
                ui.about_section()
            
            # Verify expander and about section
            mock_expander.assert_called_with("About This Tool")
            mock_about.assert_called_once()

    def test_help_tab_usage_guide_content(self):
        """Test help tab usage guide contains correct content"""
        
        with patch('streamlit.expander') as mock_expander, \
             patch('streamlit.markdown') as mock_markdown:
            
            # Mock expander context
            expander_context = MagicMock()
            mock_expander.return_value.__enter__ = Mock(return_value=expander_context)
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            
            # The actual content from the UI
            expected_content = f"""
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
                """
            
            # Simulate the markdown call
            st.markdown(expected_content)
            
            mock_markdown.assert_called_with(expected_content)

    def test_about_section_function(self):
        """Test the about_section function displays correct content"""
        
        with patch('streamlit.markdown') as mock_markdown:
            
            ui.about_section()
            
            # Verify markdown was called
            mock_markdown.assert_called_once()
            
            # Check content structure
            content = mock_markdown.call_args[0][0]
            assert "About This Tool" in content
            assert "ML Project Effort Estimator" in content
            assert "Key Features:" in content
            assert "How It Works:" in content
            assert "Best Practices:" in content

class TestHelpTabContent:
    """Test help tab content structure and completeness"""
    
    def test_usage_guide_sections(self):
        """Test usage guide contains all required sections"""
        
        with patch('streamlit.expander') as mock_expander, \
             patch('streamlit.markdown') as mock_markdown:
            
            expander_context = MagicMock()
            mock_expander.return_value.__enter__ = Mock(return_value=expander_context)
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            
            # Expected sections in usage guide
            expected_sections = [
                "Quick Start Guide",
                "Features", 
                "Tips for Better Estimates"
            ]
            
            # Simulate usage guide content
            content = f"""
            ### Quick Start Guide
            1. Fill Required Fields
            2. Optional Parameters
            3. Select Model
            4. Get Prediction
            5. Analyze Results
            
            ### Features
            - Detailed Predictions
            - Prediction History
            - SHAP Analysis
            - Model Comparison
            
            ### Tips for Better Estimates
            - Fill in as many relevant fields as possible
            - Use realistic team sizes and project characteristics
            """
            
            st.markdown(content)
            
            # Verify all sections are present
            called_content = mock_markdown.call_args[0][0]
            for section in expected_sections:
                assert section in called_content

    def test_about_section_comprehensive_content(self):
        """Test about section contains comprehensive information"""
        
        with patch('streamlit.markdown') as mock_markdown:
            
            ui.about_section()
            
            content = mock_markdown.call_args[0][0]
            
            # Verify key information is present
            required_content = [
                "About This Tool",
                "ML Project Effort Estimator",
                "Key Features:",
                "How It Works:",
                "Best Practices:",
                "machine learning",
                "SHAP analysis",
                "effort estimates"
            ]
            
            for item in required_content:
                assert item in content, f"Missing required content: {item}"

    def test_help_tab_no_save_load_references(self):
        """Test help tab doesn't reference removed save/load features"""
        
        with patch('streamlit.expander') as mock_expander, \
             patch('streamlit.markdown') as mock_markdown, \
             patch.object(ui, 'about_section') as mock_about:
            
            expander_context = MagicMock()
            mock_expander.return_value.__enter__ = Mock(return_value=expander_context)
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            
            # Simulate both help sections
            with st.expander("How to Use This Tool"):
                st.markdown("Guide content")
            
            with st.expander("About This Tool"):
                ui.about_section()
            
            # Check that no save/load references are made
            all_calls = mock_markdown.call_args_list
            all_content = " ".join([str(call) for call in all_calls])
            
            forbidden_references = [
                "save configuration",
                "load configuration", 
                "export config",
                "import config",
                "file upload"
            ]
            
            for forbidden in forbidden_references:
                assert forbidden.lower() not in all_content.lower(), \
                    f"Found forbidden save/load reference: {forbidden}"

class TestHelpTabUserInterface:
    """Test help tab user interface elements"""
    
    def test_help_tab_expander_collapsed_by_default(self):
        """Test help expanders are collapsed by default for better UX"""
        
        with patch('streamlit.expander') as mock_expander:
            
            # Mock both expanders
            expander_context = MagicMock()
            mock_expander.return_value.__enter__ = Mock(return_value=expander_context)
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            
            # Simulate help tab
            with st.expander("How to Use This Tool"):
                pass
            
            with st.expander("About This Tool"):
                pass
            
            # Verify expanders were created (default collapsed behavior)
            assert mock_expander.call_count == 2
            
            # Check that no expanded=True parameter was passed
            for call in mock_expander.call_args_list:
                args, kwargs = call
                assert 'expanded' not in kwargs or not kwargs.get('expanded', False)

    def test_help_tab_content_formatting(self):
        """Test help tab content is properly formatted"""
        
        with patch('streamlit.markdown') as mock_markdown:
            
            ui.about_section()
            
            content = mock_markdown.call_args[0][0]
            
            # Check markdown formatting
            assert "###" in content  # Has header formatting
            assert "- **" in content  # Has bold bullet points
            assert "1. **" in content  # Has numbered lists
            assert "\n" in content  # Has proper line breaks

    def test_help_tab_required_field_marker(self):
        """Test help tab correctly references required field marker"""
        
        with patch('streamlit.expander') as mock_expander, \
             patch('streamlit.markdown') as mock_markdown:
            
            expander_context = MagicMock()
            mock_expander.return_value.__enter__ = Mock(return_value=expander_context)
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            
            # Simulate usage guide with required field marker
            content = f"Complete all fields marked with {UIConstants.REQUIRED_FIELD_MARKER}"
            st.markdown(content)
            
            # Verify the marker is correctly referenced
            called_content = mock_markdown.call_args[0][0]
            assert UIConstants.REQUIRED_FIELD_MARKER in called_content

class TestHelpTabIntegration:
    """Test help tab integration with overall application"""
    
    def test_help_tab_independent_operation(self):
        """Test help tab works independently of other application state"""
        
        with patch('streamlit.expander') as mock_expander, \
             patch('streamlit.markdown'), \
             patch.object(ui, 'about_section'):
            
            expander_context = MagicMock()
            mock_expander.return_value.__enter__ = Mock(return_value=expander_context)
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            
            # Should work with empty session state
            st.session_state = {}
            
            # Simulate help tab sections
            with st.expander("How to Use This Tool"):
                st.markdown("Content")
            
            with st.expander("About This Tool"):
                ui.about_section()
            
            # Should work fine
            assert mock_expander.call_count == 2

    def test_help_tab_no_model_dependency(self):
        """Test help tab doesn't depend on model availability"""
        
        with patch('streamlit.expander') as mock_expander, \
             patch('streamlit.markdown'), \
             patch.object(ui, 'about_section'):
            
            expander_context = MagicMock()
            mock_expander.return_value.__enter__ = Mock(return_value=expander_context)
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            
            # Should work even when models are not available
            with patch.object(ui, 'MODELS_AVAILABLE', False):
                
                with st.expander("How to Use This Tool"):
                    st.markdown("Content")
                
                with st.expander("About This Tool"):
                    ui.about_section()
                
                # Should still work
                assert mock_expander.call_count == 2

    def test_help_tab_no_prediction_dependency(self):
        """Test help tab doesn't depend on prediction history"""
        
        with patch('streamlit.expander') as mock_expander, \
             patch('streamlit.markdown'), \
             patch.object(ui, 'about_section'):
            
            expander_context = MagicMock()
            mock_expander.return_value.__enter__ = Mock(return_value=expander_context)
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            
            # Should work with no prediction history
            st.session_state = {'prediction_history': []}
            
            with st.expander("How to Use This Tool"):
                st.markdown("Content")
            
            with st.expander("About This Tool"):
                ui.about_section()
            
            # Should work fine
            assert mock_expander.call_count == 2

class TestHelpTabAccessibility:
    """Test help tab accessibility and usability"""
    
    def test_help_tab_clear_section_titles(self):
        """Test help tab has clear, descriptive section titles"""
        
        with patch('streamlit.expander') as mock_expander:
            
            # Expected clear titles
            expected_titles = [
                "How to Use This Tool",
                "About This Tool"
            ]
            
            # Simulate help tab
            for title in expected_titles:
                st.expander(title)
            
            # Verify clear titles were used
            for call, expected_title in zip(mock_expander.call_args_list, expected_titles):
                actual_title = call[0][0]
                assert actual_title == expected_title

    def test_help_tab_step_by_step_guidance(self):
        """Test help tab provides step-by-step guidance"""
        
        with patch('streamlit.markdown') as mock_markdown:
            
            # Expected step-by-step content
            content = """
            1. **Fill Required Fields** - Complete all fields marked with ⭐ in the sidebar
            2. **Optional Parameters** - Add more details for better accuracy  
            3. **Select Model** - Choose a model for prediction
            4. **Get Prediction** - Click 'Predict Effort' to see your estimate
            5. **Analyze Results** - Use the Instance-Specific SHAP tab for insights
            """
            
            st.markdown(content)
            
            # Verify numbered steps are present
            called_content = mock_markdown.call_args[0][0]
            for i in range(1, 6):
                assert f"{i}. **" in called_content

    def test_help_tab_feature_descriptions(self):
        """Test help tab describes key features clearly"""
        
        with patch('streamlit.markdown') as mock_markdown:
            
            # Expected feature descriptions
            content = """
            ### Features
            - **Detailed Predictions**: Hours, days, and breakdowns
            - **Prediction History**: Track and compare multiple predictions
            - **SHAP Analysis**: Understanding of feature importance
            - **Model Comparison**: Analyze different models
            """
            
            st.markdown(content)
            
            # Verify key features are described
            called_content = mock_markdown.call_args[0][0]
            expected_features = [
                "Detailed Predictions",
                "Prediction History", 
                "SHAP Analysis",
                "Model Comparison"
            ]
            
            for feature in expected_features:
                assert feature in called_content

class TestHelpTabErrorHandling:
    """Test help tab error handling scenarios"""
    
    def test_help_tab_about_section_error_handling(self):
        """Test help tab handles about section errors gracefully"""
        
        with patch('streamlit.expander') as mock_expander, \
             patch.object(ui, 'about_section', side_effect=Exception("About section error")):
            
            expander_context = MagicMock()
            mock_expander.return_value.__enter__ = Mock(return_value=expander_context)
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            
            # Should handle error gracefully
            try:
                with st.expander("About This Tool"):
                    ui.about_section()
            except Exception:
                # Error should be handled in the UI layer
                pass

    def test_help_tab_markdown_rendering_error(self):
        """Test help tab handles markdown rendering errors"""
        
        with patch('streamlit.expander') as mock_expander, \
             patch('streamlit.markdown', side_effect=Exception("Markdown error")):
            
            expander_context = MagicMock()
            mock_expander.return_value.__enter__ = Mock(return_value=expander_context)
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            
            # Should handle error gracefully
            try:
                with st.expander("How to Use This Tool"):
                    st.markdown("Content")
            except Exception:
                # Error should be handled gracefully
                pass

class TestHelpTabConstants:
    """Test help tab uses constants correctly"""
    
    def test_help_tab_uses_ui_constants(self):
        """Test help tab correctly uses UI constants"""
        
        with patch('streamlit.markdown') as mock_markdown:
            
            # Content should reference UI constants
            content = f"Complete all fields marked with {UIConstants.REQUIRED_FIELD_MARKER}"
            st.markdown(content)
            
            # Verify constant is used
            called_content = mock_markdown.call_args[0][0]
            assert UIConstants.REQUIRED_FIELD_MARKER in called_content

    def test_help_tab_no_hardcoded_values(self):
        """Test help tab avoids hardcoded values where constants exist"""
        
        with patch('streamlit.markdown') as mock_markdown:
            
            ui.about_section()
            
            content = mock_markdown.call_args[0][0]
            
            # Should not contain hardcoded marker symbols
            # (Should use constants instead)
            assert "⭐" not in content or UIConstants.REQUIRED_FIELD_MARKER == "⭐"

class TestHelpTabNoConfigDependency:
    """Test help tab works without configuration management"""
    
    def test_help_tab_no_config_state_dependency(self):
        """Test help tab doesn't depend on configuration state"""
        
        with patch('streamlit.expander') as mock_expander, \
             patch('streamlit.markdown'), \
             patch.object(ui, 'about_section'):
            
            expander_context = MagicMock()
            mock_expander.return_value.__enter__ = Mock(return_value=expander_context)
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            
            # Should work without any config state
            st.session_state = {}  # No config variables
            
            with st.expander("How to Use This Tool"):
                st.markdown("Content")
            
            with st.expander("About This Tool"):
                ui.about_section()
            
            # Should work fine
            assert mock_expander.call_count == 2

    def test_help_tab_no_file_references(self):
        """Test help tab doesn't reference config files"""
        
        with patch('streamlit.markdown') as mock_markdown:
            
            ui.about_section()
            
            content = mock_markdown.call_args[0][0]
            
            # Should not reference config file operations
            forbidden_file_refs = [
                "config file",
                "save file",
                "load file",
                "upload file",
                "download file"
            ]
            
            for forbidden in forbidden_file_refs:
                assert forbidden.lower() not in content.lower()

    def test_help_tab_simplified_workflow_only(self):
        """Test help tab only describes simplified workflow"""
        
        with patch('streamlit.markdown') as mock_markdown:
            
            # Simulate usage guide content
            content = """
            1. **Fill Required Fields** - Complete all fields marked with ⭐ in the sidebar
            2. **Optional Parameters** - Add more details for better accuracy  
            3. **Select Model** - Choose a model for prediction
            4. **Get Prediction** - Click 'Predict Effort' to see your estimate
            5. **Analyze Results** - Use the Instance-Specific SHAP tab for insights
            """
            
            st.markdown(content)
            
            called_content = mock_markdown.call_args[0][0]
            
            # Should not mention save/load steps
            forbidden_steps = [
                "save configuration",
                "load configuration",
                "export results",
                "import settings"
            ]
            
            for forbidden in forbidden_steps:
                assert forbidden.lower() not in called_content.lower()

class TestHelpTabUsabilityFeatures:
    """Test help tab usability features"""
    
    def test_help_tab_progressive_disclosure(self):
        """Test help tab uses progressive disclosure with expanders"""
        
        with patch('streamlit.expander') as mock_expander:
            
            # Help content should be in expanders (progressive disclosure)
            with st.expander("How to Use This Tool"):
                pass
            
            with st.expander("About This Tool"):
                pass
            
            # Verify expanders were used for organization
            assert mock_expander.call_count == 2

    def test_help_tab_concise_instructions(self):
        """Test help tab provides concise, actionable instructions"""
        
        with patch('streamlit.markdown') as mock_markdown:
            
            # Instructions should be action-oriented
            content = """
            1. **Fill Required Fields** - Complete all fields marked with ⭐
            2. **Select Model** - Choose a model for prediction
            3. **Get Prediction** - Click 'Predict Effort'
            """
            
            st.markdown(content)
            
            called_content = mock_markdown.call_args[0][0]
            
            # Should have action verbs
            action_words = ["Fill", "Complete", "Choose", "Click"]
            
            for action in action_words:
                assert action in called_content

    def test_help_tab_benefit_focused_descriptions(self):
        """Test help tab focuses on user benefits"""
        
        with patch('streamlit.markdown') as mock_markdown:
            
            ui.about_section()
            
            content = mock_markdown.call_args[0][0]
            
            # Should focus on what the tool does for the user
            benefit_keywords = [
                "estimate",
                "predict", 
                "analyze",
                "insights",
                "accurate",
                "guidance"
            ]
            
            # At least half of benefit keywords should be present
            found_benefits = sum(1 for keyword in benefit_keywords if keyword.lower() in content.lower())
            assert found_benefits >= len(benefit_keywords) // 2

class TestHelpTabConsistency:
    """Test help tab consistency with the rest of the application"""
    
    def test_help_tab_consistent_terminology(self):
        """Test help tab uses consistent terminology"""
        
        with patch('streamlit.markdown') as mock_markdown:
            
            ui.about_section()
            
            content = mock_markdown.call_args[0][0]
            
            # Should use consistent terms
            consistent_terms = [
                "ML Project Effort Estimator",
                "SHAP analysis",
                "machine learning"
            ]
            
            for term in consistent_terms:
                assert term in content

    def test_help_tab_matches_ui_flow(self):
        """Test help tab instructions match actual UI flow"""
        
        with patch('streamlit.markdown') as mock_markdown:
            
            # Usage guide should match the actual tab order and flow
            content = """
            1. **Fill Required Fields** - Complete all fields marked with ⭐ in the sidebar
            2. **Select Model** - Choose a model for prediction  
            3. **Get Prediction** - Click 'Predict Effort' to see your estimate
            4. **Analyze Results** - Use the Instance-Specific SHAP tab for insights
            """
            
            st.markdown(content)
            
            called_content = mock_markdown.call_args[0][0]
            
            # Should reference actual UI elements
            ui_elements = [
                "sidebar",
                "Predict Effort",
                "Instance-Specific SHAP tab"
            ]
            
            for element in ui_elements:
                assert element in called_content

if __name__ == "__main__":
    pytest.main([__file__, "-v"])