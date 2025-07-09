# run_constants_tests.py
"""
Simple test runner to verify constants usage after making changes
Run this script to validate your hardcoded value fixes
"""

import os
import sys
import re
import ast
import inspect
from pathlib import Path

def check_hardcoded_paths():
    """Check for hardcoded file paths in source code"""
    print("=" * 60)
    print("CHECKING FOR HARDCODED PATHS")
    print("=" * 60)
    
    files_to_check = [
        'agileee/models.py',
        'agileee/ui.py', 
        'agileee/pipeline.py',
        'agileee/shap_analysis.py'
    ]
    
    hardcoded_patterns = [
        r"'data/[^']*\.csv'",      # 'data/filename.csv'
        r'"data/[^"]*\.csv"',      # "data/filename.csv"
        r"'config/[^']*\.(csv|pkl|yaml|json)'",  # 'config/filename.ext'
        r'"config/[^"]*\.(csv|pkl|yaml|json)"',  # "config/filename.ext"
        r"'models/[^']*\.pkl'",    # 'models/filename.pkl'
        r'"models/[^"]*\.pkl"',    # "models/filename.pkl"
    ]
    
    issues_found = []
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern in hardcoded_patterns:
                        matches = re.findall(pattern, line)
                        if matches:
                            issues_found.append({
                                'file': file_path,
                                'line': line_num,
                                'content': line.strip(),
                                'matches': matches
                            })
    
    if issues_found:
        print("❌ HARDCODED PATHS FOUND:")
        for issue in issues_found:
            print(f"  📁 {issue['file']}:{issue['line']}")
            print(f"     {issue['content']}")
            print(f"     Matches: {issue['matches']}")
            print()
        return False
    else:
        print("✅ No hardcoded paths found!")
        return True

def check_magic_numbers():
    """Check for magic numbers that should be constants"""
    print("=" * 60)
    print("CHECKING FOR MAGIC NUMBERS")
    print("=" * 60)
    
    files_to_check = [
        'agileee/models.py',
        'agileee/ui.py',
        'agileee/shap_analysis.py'
    ]
    
    # Magic numbers to look for (excluding common acceptable ones like 0, 1, 2)
    magic_number_patterns = [
        r'\b4\.33\b',           # weeks per month
        r'\bn_samples\s*=\s*50\b',  # SHAP sample size
        r'\bmin\(20,',          # analysis points
        r'\b10\s*<=.*<=\s*5000\b',  # prediction ranges
        r'\[:10\]',             # top N features
        r'\[:15\]',             # max features shown
    ]
    
    issues_found = []
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    # Skip comments and docstrings
                    if line.strip().startswith('#') or '"""' in line or "'''" in line:
                        continue
                        
                    for pattern in magic_number_patterns:
                        if re.search(pattern, line):
                            issues_found.append({
                                'file': file_path,
                                'line': line_num,
                                'content': line.strip(),
                                'pattern': pattern
                            })
    
    if issues_found:
        print("❌ MAGIC NUMBERS FOUND:")
        for issue in issues_found:
            print(f"  🔢 {issue['file']}:{issue['line']}")
            print(f"     {issue['content']}")
            print(f"     Pattern: {issue['pattern']}")
            print()
        return False
    else:
        print("✅ No problematic magic numbers found!")
        return True

def check_constants_usage():
    """Check that constants are actually being used"""
    print("=" * 60)
    print("CHECKING CONSTANTS USAGE")
    print("=" * 60)
    
    try:
        from agileee.constants import FileConstants, PipelineConstants, UIConstants, ValidationConstants
        
        # Check FileConstants usage
        constants_to_check = [
            ('FileConstants.CONFIG_FOLDER', 'agileee/models.py'),
            ('FileConstants.ISBSG_PREPROCESSED_FILE', 'agileee/models.py'),
            ('UIConstants.HOURS_PER_DAY', 'agileee/ui.py'),
            ('UIConstants.DAYS_PER_WEEK', 'agileee/ui.py'),
            ('PipelineConstants.TOP_N_FEATURES', 'agileee/shap_analysis.py'),
        ]
        
        all_good = True
        
        for constant, file_path in constants_to_check:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if constant in content:
                        print(f"✅ {constant} used in {file_path}")
                    else:
                        print(f"❌ {constant} NOT used in {file_path}")
                        all_good = False
            else:
                print(f"⚠️  File not found: {file_path}")
        
        return all_good
        
    except ImportError as e:
        print(f"❌ Could not import constants: {e}")
        return False

def check_file_existence():
    """Check that expected files exist"""
    print("=" * 60)
    print("CHECKING FILE EXISTENCE")
    print("=" * 60)
    
    try:
        from agileee.constants import FileConstants
        
        expected_files = [
            (os.path.join(FileConstants.CONFIG_FOLDER, FileConstants.UI_INFO_FILE), "UI config"),
            (os.path.join(FileConstants.CONFIG_FOLDER, FileConstants.FEATURE_MAPPING_FILE), "Feature mapping"),
            (os.path.join(FileConstants.CONFIG_FOLDER, FileConstants.ISBSG_PREPROCESSED_FILE), "ISBSG data"),
        ]
        
        all_exist = True
        
        for file_path, description in expected_files:
            if os.path.exists(file_path):
                print(f"✅ {description}: {file_path}")
            else:
                print(f"❌ MISSING {description}: {file_path}")
                all_exist = False
        
        return all_exist
        
    except ImportError:
        print("❌ Could not import FileConstants")
        return False

def test_actual_functionality():
    """Test that key functions work with constants"""
    print("=" * 60)
    print("TESTING ACTUAL FUNCTIONALITY")
    print("=" * 60)
    
    try:
        # Test 1: Import modules
        from agileee import models, constants
        print("✅ Module imports successful")
        
        # Test 2: Constants are accessible
        config_folder = constants.FileConstants.CONFIG_FOLDER
        print(f"✅ CONFIG_FOLDER: {config_folder}")
        
        # Test 3: File path construction
        test_path = os.path.join(
            constants.FileConstants.CONFIG_FOLDER,
            constants.FileConstants.UI_INFO_FILE
        )
        print(f"✅ Path construction: {test_path}")
        
        # Test 4: Time calculation constants
        hours_per_day = constants.UIConstants.HOURS_PER_DAY
        days_per_week = constants.UIConstants.DAYS_PER_WEEK
        weeks_per_month = constants.UIConstants.WEEKS_PER_MONTH
        print(f"✅ Time constants: {hours_per_day}h/day, {days_per_week}d/week, {weeks_per_month}w/month")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Run all checks"""
    print("CONSTANTS USAGE VERIFICATION")
    print("=" * 60)
    print("This script checks if hardcoded values have been replaced with constants")
    print()
    
    results = []
    
    # Run all checks
    results.append(("Hardcoded Paths", check_hardcoded_paths()))
    results.append(("Magic Numbers", check_magic_numbers()))
    results.append(("Constants Usage", check_constants_usage()))
    results.append(("File Existence", check_file_existence()))
    results.append(("Functionality", test_actual_functionality()))
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 ALL CHECKS PASSED! Constants are properly implemented.")
    else:
        print("⚠️  SOME CHECKS FAILED. Please review the issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)