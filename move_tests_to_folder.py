# move_tests_to_folder.py
"""
Script to help organize test files into the proper /tests folder structure.
Run this script to automatically create the folder structure and move test files.
"""

import os
import shutil
from pathlib import Path

def create_test_structure():
    """Create the test folder structure."""
    
    test_dirs = [
        "tests",
        "tests/tabs",
        "tests/unit", 
        "tests/integration",
        "tests/fixtures"
    ]
    
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files to make them Python packages
        init_file = Path(dir_path) / "__init__.py"
        if not init_file.exists():
            init_file.write_text("# Test package\n")
    
    print("✅ Test directory structure created")

def create_init_files():
    """Create __init__.py files for test packages."""
    
    init_files = [
        "tests/__init__.py",
        "tests/tabs/__init__.py", 
        "tests/unit/__init__.py",
        "tests/integration/__init__.py",
        "tests/fixtures/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).write_text("# Test package\n")
    
    print("✅ __init__.py files created")

def move_test_files():
    """Move test files to appropriate directories if they exist in current directory."""
    
    # Mapping of current test files to their new locations
    file_mappings = {
        # Tab tests
        "test_estimator_tab.py": "tests/tabs/test_estimator_tab.py",
        "test_shap_tab.py": "tests/tabs/test_shap_tab.py", 
        "test_model_comparison_tab.py": "tests/tabs/test_model_comparison_tab.py",
        "test_static_shap_tab.py": "tests/tabs/test_static_shap_tab.py",
        "test_help_tab.py": "tests/tabs/test_help_tab.py",
        
        # Unit tests
        "test_removed_functions.py": "tests/unit/test_removed_functions.py",
        "test_ui_simplification.py": "tests/unit/test_ui_functions.py",
        
        # Integration tests
        "test_cross_tab_integration.py": "tests/integration/test_cross_tab_integration.py",
        
        # Main test runner
        "run_all_ui_tests.py": "tests/run_all_tests.py"
    }
    
    moved_files = []
    missing_files = []
    
    for source, destination in file_mappings.items():
        if Path(source).exists():
            # Create destination directory if it doesn't exist
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            
            # Move the file
            shutil.move(source, destination)
            moved_files.append(f"{source} → {destination}")
        else:
            missing_files.append(source)
    
    if moved_files:
        print("✅ Files moved:")
        for move in moved_files:
            print(f"   {move}")
    
    if missing_files:
        print("\n⚠️  Files not found (will need to be created manually):")
        for missing in missing_files:
            print(f"   {missing}")

def update_imports_in_test_files():
    """Update import statements in test files to work from tests directory."""
    
    test_files = [
        "tests/tabs/test_estimator_tab.py",
        "tests/tabs/test_shap_tab.py",
        "tests/tabs/test_model_comparison_tab.py", 
        "tests/tabs/test_static_shap_tab.py",
        "tests/tabs/test_help_tab.py",
        "tests/unit/test_removed_functions.py"
    ]
    
    for test_file in test_files:
        file_path = Path(test_file)
        if file_path.exists():
            try:
                content = file_path.read_text()
                
                # Replace the old path insertion with new one
                old_path_line = "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))"
                new_path_line = "sys.path.insert(0, str(Path(__file__).parent.parent.parent))"
                
                if old_path_line in content:
                    content = content.replace(old_path_line, new_path_line)
                    # Add Path import if not present
                    if "from pathlib import Path" not in content:
                        content = content.replace(
                            "import sys\nimport os", 
                            "import sys\nimport os\nfrom pathlib import Path"
                        )
                    
                    file_path.write_text(content)
                    print(f"✅ Updated imports in {test_file}")
                
            except Exception as e:
                print(f"⚠️  Could not update {test_file}: {e}")

def create_missing_test_files():
    """Create any missing essential test files."""
    
    # Create fixtures file
    fixtures_file = Path("tests/fixtures/mock_data.py")
    if not fixtures_file.exists():
        fixtures_content = '''"""
Mock data and fixtures for UI tests.
"""

import pandas as pd
import numpy as np

def get_sample_user_inputs():
    """Get sample user inputs for testing."""
    return {
        'project_prf_functional_size': 100,
        'project_prf_max_team_size': 5,
        'external_eef_industry_sector': 'Financial',
        'tech_tf_primary_programming_language': 'Java',
        'selected_model': 'rf_model'
    }

def get_sample_prediction_history():
    """Get sample prediction history."""
    return [
        {
            'timestamp': '2024-01-01 10:00:00',
            'model': 'Random Forest',
            'model_technical': 'rf_model', 
            'prediction_hours': 480.0,
            'inputs': get_sample_user_inputs()
        }
    ]
'''
        fixtures_file.write_text(fixtures_content)
        print("✅ Created tests/fixtures/mock_data.py")

    # Create integration test file
    integration_file = Path("tests/integration/test_cross_tab_integration.py")
    if not integration_file.exists():
        integration_content = '''"""
Integration tests for cross-tab functionality.
"""

import pytest
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import ui

class TestCrossTabIntegration:
    """Test integration between different tabs."""
    
    def test_estimator_to_shap_flow(self):
        """Test that predictions flow from estimator to SHAP tab."""
        # Basic integration test
        assert hasattr(ui, 'main')
        assert hasattr(ui, 'display_instance_specific_shap')
'''
        integration_file.write_text(integration_content)
        print("✅ Created tests/integration/test_cross_tab_integration.py")

def main():
    """Main function to set up test structure."""
    
    print("🧪 Setting up UI Test Structure")
    print("="*50)
    
    # Step 1: Create directory structure
    print("\n1. Creating test directory structure...")
    create_test_structure()
    
    # Step 2: Create __init__.py files
    print("\n2. Creating package files...")
    create_init_files()
    
    # Step 3: Move existing test files
    print("\n3. Moving existing test files...")
    move_test_files()
    
    # Step 4: Update imports
    print("\n4. Updating import statements...")
    update_imports_in_test_files()
    
    # Step 5: Create missing files
    print("\n5. Creating missing test files...")
    create_missing_test_files()
    
    print("\n" + "="*50)
    print("✅ Test structure setup complete!")
    print("\nNext steps:")
    print("1. Copy the test files from the artifacts into the appropriate directories")
    print("2. Install test dependencies: pip install -r tests/requirements-test.txt")
    print("3. Run tests: cd tests && python run_all_tests.py --quick")
    print("\nTest structure:")
    print("tests/")
    print("├── conftest.py              # Shared fixtures")
    print("├── pytest.ini              # Pytest config") 
    print("├── requirements-test.txt    # Test dependencies")
    print("├── run_all_tests.py        # Main test runner")
    print("├── tabs/                   # Tab-specific tests")
    print("├── unit/                   # Unit tests")
    print("├── integration/            # Integration tests")
    print("└── fixtures/               # Test data")

if __name__ == "__main__":
    main()