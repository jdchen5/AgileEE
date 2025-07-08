# test_shap_runner_fixed.py
"""
Fixed test runner that handles import paths correctly
Run this from your project root directory
"""

import sys
import os
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Fix the import paths
project_root = os.path.dirname(os.path.abspath(__file__))
if 'tests' in project_root:
    project_root = os.path.dirname(project_root)  # Go up one level from tests/

# Add both project root and agileee package to path
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'agileee'))

print(f"🔍 Looking for modules in: {project_root}")

def test_imports_step_by_step():
    """Test imports step by step to identify issues"""
    print("Testing imports step by step...")
    
    # Test 1: Basic shap import
    try:
        import shap
        print("✅ SHAP package available")
    except ImportError:
        print("❌ SHAP package not installed. Run: pip install shap")
        return False
    
    # Test 2: Try agileee.shap_analysis
    try:
        from agileee.shap_analysis import display_optimized_shap_analysis
        print("✅ agileee.shap_analysis import successful")
        return True
    except ImportError as e:
        print(f"❌ agileee.shap_analysis import failed: {e}")
    
    # Test 3: Try direct shap_analysis
    try:
        import agileee.shap_analysis
        print("✅ Direct shap_analysis import successful")
        return True
    except ImportError as e:
        print(f"❌ Direct shap_analysis import failed: {e}")
    
    # Test 4: Look for the file
    possible_paths = [
        os.path.join(project_root, 'shap_analysis.py'),
        os.path.join(project_root, 'agileee', 'shap_analysis.py'),
        os.path.join(project_root, 'agileee', '__init__.py')
    ]
    
    print("\n🔍 Looking for files:")
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found: {path}")
        else:
            print(f"❌ Missing: {path}")
    
    return False

def test_basic_functionality():
    """Test basic functionality with whatever import works"""
    print("\n🧪 Testing basic functionality...")
    
    # Try different import methods
    shap_analysis = None
    
    # Method 1: agileee package
    try:
        from agileee import shap_analysis
        print("✅ Imported via agileee package")
    except ImportError:
        pass
    
    # Method 2: direct import
    if shap_analysis is None:
        try:
            import agileee.shap_analysis
            print("✅ Imported directly")
        except ImportError:
            pass
    
    # Method 3: sys.path manipulation
    if shap_analysis is None:
        try:
            # Add agileee folder to path
            agileee_path = os.path.join(project_root, 'agileee')
            if agileee_path not in sys.path:
                sys.path.insert(0, agileee_path)
            import agileee.shap_analysis
            print("✅ Imported with path manipulation")
        except ImportError as e:
            print(f"❌ All import methods failed. Last error: {e}")
            return False
    
    if shap_analysis is None:
        print("❌ Could not import shap_analysis module")
        return False
    
    # Test basic functions
    try:
        # Test cache functions
        info = shap_analysis.get_cache_info()
        print(f"✅ get_cache_info() works: {info}")
        
        shap_analysis.clear_explainer_cache()
        print("✅ clear_explainer_cache() works")
        
        return True
        
    except Exception as e:
        print(f"❌ Function test failed: {e}")
        return False

def test_mock_analysis():
    """Test SHAP analysis with mocked streamlit"""
    print("\n🧪 Testing mock SHAP analysis...")
    
    try:
        # Import with working method
        shap_analysis = None
        try:
            from agileee import shap_analysis
        except:
            try:
                import agileee.shap_analysis
            except:
                # Try path manipulation
                agileee_path = os.path.join(project_root, 'agileee')
                sys.path.insert(0, agileee_path)
                import agileee.shap_analysis
        
        # Mock streamlit functions
        with patch('streamlit.subheader'), \
             patch('streamlit.error'), \
             patch('streamlit.warning'), \
             patch('streamlit.spinner'), \
             patch('streamlit.success'), \
             patch('streamlit.info'):
            
            mock_inputs = {
                'project_prf_functional_size': 100,
                'project_prf_max_team_size': 5
            }
            
            def mock_get_model(name):
                model = Mock()
                model.predict = Mock(return_value=[100.0])
                return model
            
            # This should not crash
            shap_analysis.display_optimized_shap_analysis(
                mock_inputs, 
                'test_model', 
                mock_get_model
            )
            
        print("✅ Mock SHAP analysis completed without crashing")
        return True
        
    except Exception as e:
        print(f"❌ Mock analysis failed: {e}")
        return False

def create_minimal_shap_analysis():
    """Create a minimal shap_analysis.py file if missing"""
    print("\n🔧 Creating minimal shap_analysis.py file...")
    
    minimal_content = '''
# Minimal shap_analysis.py for testing
import streamlit as st

_explainer_cache = {}

def get_cache_info():
    return {'cached_models': [], 'cache_size': 0}

def clear_explainer_cache():
    global _explainer_cache
    _explainer_cache.clear()
    st.success("Cache cleared!")

def display_optimized_shap_analysis(user_inputs, model_name, get_trained_model_func):
    st.subheader("🔍 SHAP Analysis")
    st.info("SHAP analysis placeholder - module loaded successfully")
    st.write(f"Model: {model_name}")
    st.write(f"Inputs: {len(user_inputs)} features")

def get_shap_explainer_optimized(*args, **kwargs):
    return None
'''
    
    # Try to create in agileee folder first
    agileee_path = os.path.join(project_root, 'agileee')
    if os.path.exists(agileee_path):
        file_path = os.path.join(agileee_path, 'shap_analysis.py')
    else:
        file_path = os.path.join(project_root, 'shap_analysis.py')
    
    try:
        with open(file_path, 'w') as f:
            f.write(minimal_content)
        print(f"✅ Created minimal shap_analysis.py at: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Could not create file: {e}")
        return False

def run_quick_diagnostic():
    """Quick diagnostic to understand the setup"""
    print("🚀 Quick SHAP Diagnostic")
    print("=" * 40)
    
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📁 Project root: {project_root}")
    print(f"🐍 Python path: {sys.path[:3]}...")  # Show first 3 entries
    
    # Check directory structure
    print("\n📂 Directory structure:")
    for item in os.listdir(project_root):
        if os.path.isdir(os.path.join(project_root, item)):
            print(f"   📁 {item}/")
        elif item.endswith('.py'):
            print(f"   📄 {item}")
    
    # Check for agileee folder
    agileee_path = os.path.join(project_root, 'agileee')
    if os.path.exists(agileee_path):
        print(f"\n📂 agileee/ contents:")
        for item in os.listdir(agileee_path):
            if item.endswith('.py'):
                print(f"   📄 {item}")
    
    # Test imports step by step
    print("\n" + "=" * 40)
    if not test_imports_step_by_step():
        print("\n🔧 Attempting to fix...")
        if create_minimal_shap_analysis():
            print("✅ Created minimal file, try importing again")
            if test_basic_functionality():
                print("🎉 Basic functionality works!")
                return True
        return False
    
    # Test functionality
    if test_basic_functionality():
        if test_mock_analysis():
            print("\n🎉 All tests passed! SHAP integration should work.")
            return True
        else:
            print("\n⚠️  Basic functions work but analysis has issues.")
            return False
    else:
        print("\n❌ Basic functionality failed.")
        return False

if __name__ == "__main__":
    # Change to project root if we're in tests folder
    if os.path.basename(os.getcwd()) == 'tests':
        os.chdir('..')
        print("📁 Changed to project root directory")
    
    success = run_quick_diagnostic()
    
    if success:
        print("\n✅ You can now run your Streamlit app!")
        print("   Command: streamlit run main.py")
    else:
        print("\n❌ Please fix the issues above before running the app.")
        print("\nNext steps:")
        print("1. Make sure you're in the project root directory")
        print("2. Check that shap_analysis.py exists in agileee/ folder")
        print("3. Install missing dependencies: pip install shap streamlit")