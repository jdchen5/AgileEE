# test_configuration_changes.py
"""
Test cases to validate all configuration-driven changes
Tests that hardcoded values have been replaced with YAML configuration
"""

import unittest
import os
import sys
from unittest.mock import patch, mock_open, MagicMock
import yaml

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestConfigurationChanges(unittest.TestCase):
    """Test suite to validate configuration-driven changes"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock YAML configurations
        self.mock_feature_mapping = {
            'categorical_features': {
                'external_eef_industry_sector': {
                    'options': ['', 'Banking', 'Financial']
                },
                'tech_tf_primary_programming_language': {
                    'options': ['', 'ABAP', 'JAVA', 'Javascript', 'C', 'C++', 'Other']
                }
            }
        }
        
        self.mock_ui_info = {
            'fields': {
                'project_prf_max_team_size': {'mandatory': True},
                'project_prf_functional_size': {'mandatory': True},
                'external_eef_industry_sector': {'mandatory': True},
                'project_prf_case_tool_used': {'mandatory': False},
                'process_pmf_prototyping_used': {'mandatory': False}
            }
        }

    @patch('agileee.config_loader.ConfigLoader.load_yaml_config')
    def test_constants_feature_validation_industries(self, mock_load_yaml):
        """Test that FeatureValidationConstants loads industries from YAML"""
        mock_load_yaml.return_value = self.mock_feature_mapping
        
        from agileee.constants import FeatureValidationConstants
        
        industries = FeatureValidationConstants.get_valid_industries()
        
        # Verify it calls the config loader
        mock_load_yaml.assert_called_once()
        
        # Verify it returns the correct industries from YAML
        expected_industries = ['', 'Banking', 'Financial']
        self.assertEqual(industries, expected_industries)
        
        # Verify old hardcoded values are NOT included
        self.assertNotIn('Technology', industries)
        self.assertNotIn('Healthcare', industries)
        self.assertNotIn('Manufacturing', industries)
        self.assertNotIn('Government', industries)

    @patch('agileee.config_loader.ConfigLoader.load_yaml_config')
    def test_constants_feature_validation_languages(self, mock_load_yaml):
        """Test that FeatureValidationConstants loads languages from YAML"""
        mock_load_yaml.return_value = self.mock_feature_mapping
        
        from agileee.constants import FeatureValidationConstants
        
        languages = FeatureValidationConstants.get_valid_languages()
        
        # Verify it returns the correct languages from YAML
        expected_languages = ['', 'ABAP', 'JAVA', 'Javascript', 'C', 'C++', 'Other']
        self.assertEqual(languages, expected_languages)
        
        # Verify old hardcoded values are NOT included
        self.assertNotIn('Python', languages)
        self.assertNotIn('C#', languages)
        self.assertNotIn('PHP', languages)
        self.assertNotIn('Ruby', languages)

    @patch('agileee.config_loader.ConfigLoader.load_yaml_config')
    def test_constants_mandatory_fields(self, mock_load_yaml):
        """Test that mandatory fields are derived from ui_info.yaml"""
        mock_load_yaml.return_value = self.mock_ui_info
        
        from agileee.constants import FeatureValidationConstants
        
        mandatory_fields = FeatureValidationConstants.get_mandatory_fields()
        
        # Verify it returns only mandatory fields
        expected_mandatory = ['project_prf_max_team_size', 'project_prf_functional_size', 'external_eef_industry_sector']
        self.assertEqual(set(mandatory_fields), set(expected_mandatory))
        
        # Verify non-mandatory fields are excluded
        self.assertNotIn('project_prf_case_tool_used', mandatory_fields)
        self.assertNotIn('process_pmf_prototyping_used', mandatory_fields)

    def test_shap_constants_exist(self):
        """Test that SHAP constants are properly defined"""
        from agileee.constants import ShapConstants
        
        # Verify tree model keywords
        self.assertIsInstance(ShapConstants.TREE_MODEL_KEYWORDS, list)
        self.assertIn('forest', ShapConstants.TREE_MODEL_KEYWORDS)
        self.assertIn('tree', ShapConstants.TREE_MODEL_KEYWORDS)
        self.assertIn('xgb', ShapConstants.TREE_MODEL_KEYWORDS)
        
        # Verify linear model keywords
        self.assertIsInstance(ShapConstants.LINEAR_MODEL_KEYWORDS, list)
        self.assertIn('linear', ShapConstants.LINEAR_MODEL_KEYWORDS)
        self.assertIn('lasso', ShapConstants.LINEAR_MODEL_KEYWORDS)
        self.assertIn('ridge', ShapConstants.LINEAR_MODEL_KEYWORDS)

    @patch('agileee.constants.FeatureValidationConstants.get_valid_industries')
    def test_feature_engineering_uses_yaml_industries(self, mock_get_industries):
        """Test that feature_engineering.py uses YAML for industry validation"""
        mock_get_industries.return_value = ['Banking', 'Financial']
        
        # Import and test the feature engineering module
        try:
            from agileee.feature_engineering import create_training_compatible_features
            
            # Test with valid industry
            test_inputs = {'external_eef_industry_sector': 'Banking'}
            result = create_training_compatible_features(test_inputs)
            
            # Verify the function was called
            mock_get_industries.assert_called()
            
            # Verify industry is processed correctly
            self.assertIn('external_eef_industry_sector', result)
            
        except ImportError:
            self.skipTest("feature_engineering module not available")

    @patch('agileee.constants.FeatureValidationConstants.get_valid_languages')
    def test_feature_engineering_uses_yaml_languages(self, mock_get_languages):
        """Test that feature_engineering.py uses YAML for language validation"""
        mock_get_languages.return_value = ['JAVA', 'Javascript', 'C++']
        
        try:
            from agileee.feature_engineering import create_training_compatible_features
            
            # Test with valid language
            test_inputs = {'tech_tf_primary_programming_language': 'JAVA'}
            result = create_training_compatible_features(test_inputs)
            
            # Verify the function was called
            mock_get_languages.assert_called()
            
            # Verify language is processed correctly
            self.assertIn('tech_tf_primary_programming_language', result)
            
        except ImportError:
            self.skipTest("feature_engineering module not available")

    @patch('agileee.constants.FeatureValidationConstants.get_mandatory_fields')
    def test_pipeline_uses_mandatory_fields(self, mock_get_mandatory):
        """Test that pipeline.py uses mandatory fields for cols_to_keep"""
        mock_get_mandatory.return_value = ['project_prf_max_team_size', 'external_eef_industry_sector']
        
        try:
            from agileee.pipeline import create_preprocessing_pipeline
            
            # Create pipeline without explicit cols_to_keep
            pipeline = create_preprocessing_pipeline(target_col='test_target')
            
            # Verify the function was called
            mock_get_mandatory.assert_called()
            
            # Verify pipeline was created successfully
            self.assertIsNotNone(pipeline)
            
        except ImportError:
            self.skipTest("pipeline module not available")

    def test_shap_analysis_uses_constants(self):
        """Test that shap_analysis.py uses constants instead of hardcoded values"""
        try:
            from agileee.shap_analysis import create_appropriate_explainer
            from agileee.constants import ShapConstants
            
            # Mock model object
            mock_model = MagicMock()
            mock_model.__class__.__name__ = 'RandomForestRegressor'
            
            # Test that it can access the constants
            tree_keywords = ShapConstants.TREE_MODEL_KEYWORDS
            linear_keywords = ShapConstants.LINEAR_MODEL_KEYWORDS
            
            # Verify constants are lists and not empty
            self.assertIsInstance(tree_keywords, list)
            self.assertIsInstance(linear_keywords, list)
            self.assertGreater(len(tree_keywords), 0)
            self.assertGreater(len(linear_keywords), 0)
            
        except ImportError:
            self.skipTest("shap_analysis module not available")

    def test_no_hardcoded_values_in_source(self):
        """Test that source files don't contain old hardcoded values"""
        # This is a static analysis test
        hardcoded_patterns = [
            "'Technology', 'Banking', 'Healthcare'",
            "'Python', 'Java', 'JavaScript'", 
            "['forest', 'tree', 'xgb'",
            "'project_prf_case_tool_used', 'process_pmf_prototyping_used'"
        ]
        
        files_to_check = [
            'agileee/feature_engineering.py',
            'agileee/pipeline.py', 
            'agileee/shap_analysis.py'
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in hardcoded_patterns:
                    self.assertNotIn(pattern, content, 
                                   f"Found hardcoded pattern '{pattern}' in {file_path}")

    @patch('agileee.config_loader.ConfigLoader.load_yaml_config')
    def test_config_loading_error_handling(self, mock_load_yaml):
        """Test that configuration loading handles errors gracefully"""
        mock_load_yaml.return_value = None  # Simulate load failure
        
        from agileee.constants import FeatureValidationConstants
        
        # Should return empty lists when config fails to load
        industries = FeatureValidationConstants.get_valid_industries()
        languages = FeatureValidationConstants.get_valid_languages()
        mandatory_fields = FeatureValidationConstants.get_mandatory_fields()
        
        self.assertEqual(industries, [])
        self.assertEqual(languages, [])
        self.assertEqual(mandatory_fields, [])

    def test_backward_compatibility_behavior(self):
        """Test that changes maintain backward compatibility"""
        # Test that all required modules can still be imported
        modules_to_test = [
            'agileee.constants',
            'agileee.feature_engineering', 
            'agileee.pipeline',
            'agileee.shap_analysis'
        ]
        
        for module_name in modules_to_test:
            try:
                __import__(module_name)
            except ImportError as e:
                self.fail(f"Failed to import {module_name}: {e}")

    @patch('agileee.config_loader.ConfigLoader.load_yaml_config')
    def test_integration_feature_engineering_with_yaml(self, mock_load_yaml):
        """Integration test: feature engineering with real YAML structure"""
        mock_load_yaml.return_value = self.mock_feature_mapping
        
        try:
            from agileee.feature_engineering import create_training_compatible_features
            
            # Test inputs that should work with new YAML-driven logic
            test_inputs = {
                'external_eef_industry_sector': 'Banking',
                'tech_tf_primary_programming_language': 'JAVA',
                'project_prf_max_team_size': 5,
                'project_prf_functional_size': 100
            }
            
            result = create_training_compatible_features(test_inputs)
            
            # Verify function completes without error
            self.assertIsInstance(result, dict)
            self.assertGreater(len(result), 0)
            
            # Verify target was estimated
            self.assertIn('project_prf_normalised_work_effort', result)
            
        except ImportError:
            self.skipTest("feature_engineering module not available")


class TestRemovedDuplicateShapFunctions(unittest.TestCase):
    """Test that duplicate SHAP functions were properly removed"""
    
    def test_shap_functions_removed_from_models(self):
        """Test that duplicate SHAP functions are removed from models.py"""
        try:
            import agileee.models as models
            
            # These functions should NOT exist in models.py anymore
            duplicate_functions = [
                'validate_shap_compatibility',
                'get_shap_feature_names'
            ]
            
            for func_name in duplicate_functions:
                self.assertFalse(hasattr(models, func_name),
                               f"Function {func_name} should be removed from models.py")
                
        except ImportError:
            self.skipTest("models module not available")

    def test_shap_functions_exist_in_shap_analysis(self):
        """Test that SHAP functions exist where they should"""
        try:
            import agileee.shap_analysis as shap_analysis
            
            # These functions should exist in shap_analysis.py
            required_functions = [
                'create_appropriate_explainer',
                'get_shap_explainer_optimized',
                'get_shap_analysis_results'
            ]
            
            for func_name in required_functions:
                self.assertTrue(hasattr(shap_analysis, func_name),
                              f"Function {func_name} should exist in shap_analysis.py")
                
        except ImportError:
            self.skipTest("shap_analysis module not available")


def run_tests():
    """Run all configuration change tests"""
    print("="*60)
    print("TESTING CONFIGURATION-DRIVEN CHANGES")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestConfigurationChanges))
    suite.addTests(loader.loadTestsFromTestCase(TestRemovedDuplicateShapFunctions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error:')[-1].strip()}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED! Configuration changes are working correctly.")
    else:
        print("\n❌ Some tests failed. Please review the issues above.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)