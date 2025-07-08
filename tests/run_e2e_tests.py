#!/usr/bin/env python3
"""
End-to-End Test Runner for AgileEE
Runs comprehensive end-to-end tests and generates detailed reports.
"""

import sys
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

def print_banner():
    """Print test runner banner"""
    print("=" * 80)
    print("🧪 AgileEE End-to-End Test Suite")
    print("=" * 80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Working Directory: {os.getcwd()}")
    print("=" * 80)

def print_section(title):
    """Print section header"""
    print(f"\n🔍 {title}")
    print("-" * 60)

def run_test_file(test_file, verbose=True):
    """Run a specific test file and return results"""
    print(f"Running: {test_file}")
    
    cmd = ["python", "-m", "pytest", test_file, "-v"] if verbose else ["python", "-m", "pytest", test_file]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        duration = end_time - start_time
        
        success = result.returncode == 0
        
        if success:
            print(f"✅ PASSED ({duration:.2f}s)")
        else:
            print(f"❌ FAILED ({duration:.2f}s)")
            if result.stdout:
                print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            if result.stderr:
                print("STDERR:", result.stderr[-500:])  # Last 500 chars
        
        return {
            'file': test_file,
            'success': success,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
        
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT ({test_file})")
        return {
            'file': test_file,
            'success': False,
            'duration': 300.0,
            'stdout': '',
            'stderr': 'Test timed out after 300 seconds',
            'returncode': -1
        }
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return {
            'file': test_file,
            'success': False,
            'duration': 0.0,
            'stdout': '',
            'stderr': str(e),
            'returncode': -2
        }

def check_prerequisites():
    """Check if prerequisites are met"""
    print_section("Prerequisites Check")
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required, found {sys.version}")
    else:
        print("✅ Python version OK")
    
    # Check if we're in the right directory
    if not os.path.exists("agileee") and not os.path.exists("ui.py"):
        issues.append("Not in project root directory (missing agileee/ or ui.py)")
    else:
        print("✅ Project structure OK")
    
    # Check required packages
    required_packages = ["pytest", "streamlit", "pandas", "numpy"]
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            issues.append(f"Missing required package: {package}")
    
    # Check test files exist
    test_files = [
        "tests/test_e2e_complete_workflow.py",
        "tests/test_e2e_user_scenarios.py", 
        "tests/test_e2e_system_integration.py"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"✅ {test_file} found")
        else:
            issues.append(f"Missing test file: {test_file}")
    
    if issues:
        print("\n❌ Prerequisites not met:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    
    print("\n✅ All prerequisites met!")
    return True

def run_all_e2e_tests():
    """Run all end-to-end tests"""
    print_section("Running End-to-End Tests")
    
    # Define test files in order of execution
    test_files = [
        "tests/test_e2e_complete_workflow.py",
        "tests/test_e2e_user_scenarios.py",
        "tests/test_e2e_system_integration.py"
    ]
    
    results = []
    total_start = time.time()
    
    for test_file in test_files:
        if os.path.exists(test_file):
            result = run_test_file(test_file)
            results.append(result)
        else:
            print(f"⚠️  Skipping missing file: {test_file}")
            results.append({
                'file': test_file,
                'success': False,
                'duration': 0.0,
                'stdout': '',
                'stderr': 'File not found',
                'returncode': -3
            })
    
    total_duration = time.time() - total_start
    
    return results, total_duration

def run_additional_tests():
    """Run existing component tests as well"""
    print_section("Running Additional Component Tests")
    
    additional_tests = [
        "tests/test_model_comparison_tab.py",
        "tests/test_estimator_tab.py",
        "tests/test_help_tab.py",
        "tests/test_static_shap_tab.py"
    ]
    
    results = []
    
    for test_file in additional_tests:
        if os.path.exists(test_file):
            result = run_test_file(test_file, verbose=False)  # Less verbose for component tests
            results.append(result)
        else:
            print(f"⚠️  Skipping missing file: {test_file}")
    
    return results

def generate_report(e2e_results, additional_results, total_duration):
    """Generate comprehensive test report"""
    print_section("Test Results Summary")
    
    all_results = e2e_results + additional_results
    
    # Calculate statistics
    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results if r['success'])
    failed_tests = total_tests - passed_tests
    
    total_test_duration = sum(r['duration'] for r in all_results)
    
    print(f"📊 Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"⏱️  Total Duration: {total_duration:.2f}s")
    print(f"🧪 Test Execution Time: {total_test_duration:.2f}s")
    print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    # Detailed results
    print("\n📋 Detailed Results:")
    print("-" * 60)
    
    for result in all_results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        filename = os.path.basename(result['file'])
        print(f"{status} {filename:<30} ({result['duration']:.2f}s)")
        
        if not result['success'] and result['stderr']:
            print(f"    Error: {result['stderr'][:100]}...")
    
    # Save detailed report to file
    save_detailed_report(all_results, total_duration)
    
    return passed_tests == total_tests

def save_detailed_report(results, total_duration):
    """Save detailed report to file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"e2e_test_report_{timestamp}.txt"
    
    try:
        with open(report_file, 'w') as f:
            f.write("AgileEE End-to-End Test Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Duration: {total_duration:.2f}s\n\n")
            
            for result in results:
                f.write(f"Test File: {result['file']}\n")
                f.write(f"Status: {'PASS' if result['success'] else 'FAIL'}\n")
                f.write(f"Duration: {result['duration']:.2f}s\n")
                f.write(f"Return Code: {result['returncode']}\n")
                
                if result['stdout']:
                    f.write("STDOUT:\n")
                    f.write(result['stdout'])
                    f.write("\n")
                
                if result['stderr']:
                    f.write("STDERR:\n")
                    f.write(result['stderr'])
                    f.write("\n")
                
                f.write("-" * 50 + "\n")
        
        print(f"\n📄 Detailed report saved: {report_file}")
        
    except Exception as e:
        print(f"⚠️  Could not save report: {e}")

def run_quick_smoke_tests():
    """Run quick smoke tests to verify basic functionality"""
    print_section("Quick Smoke Tests")
    
    try:
        # Test 1: Import main modules
        print("Testing imports...")
        sys.path.insert(0, '.')
        
        try:
            import agileee.ui as ui
            print("✅ UI module import OK")
        except ImportError as e:
            print(f"❌ UI module import failed: {e}")
            return False
        
        try:
            from agileee.constants import UIConstants, FileConstants
            print("✅ Constants import OK")
        except ImportError as e:
            print(f"❌ Constants import failed: {e}")
            return False
        
        # Test 2: Basic function availability
        print("Testing basic functions...")
        
        essential_functions = [
            'initialize_session_state',
            'sidebar_inputs', 
            'display_model_comparison',
            'show_prediction',
            'about_section'
        ]
        
        for func_name in essential_functions:
            if hasattr(ui, func_name):
                print(f"✅ {func_name} available")
            else:
                print(f"❌ {func_name} missing")
                return False
        
        # Test 3: Session state initialization
        print("Testing session state...")
        try:
            import streamlit as st
            st.session_state = {}  # Reset for test
            ui.initialize_session_state()
            
            required_keys = ['prediction_history', 'current_prediction_results']
            for key in required_keys:
                if key in st.session_state:
                    print(f"✅ {key} initialized")
                else:
                    print(f"❌ {key} not initialized")
                    return False
                    
        except Exception as e:
            print(f"❌ Session state test failed: {e}")
            return False
        
        print("✅ All smoke tests passed!")
        return True
        
    except Exception as e:
        print(f"💥 Smoke tests failed: {e}")
        return False

def run_configuration_tests():
    """Test configuration loading"""
    print_section("Configuration Tests")
    
    try:
        # Check for config files
        config_files = [
            "config/ui_info.yaml",
            "config/feature_mapping.yaml"
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                print(f"✅ {config_file} found")
            else:
                print(f"⚠️  {config_file} missing (using defaults)")
        
        # Test configuration loading
        try:
            import agileee.ui as ui
            
            # Test field configuration
            if hasattr(ui, 'FIELDS') and ui.FIELDS:
                print("✅ Field configuration loaded")
            else:
                print("⚠️  No field configuration found")
            
            # Test tab organization
            if hasattr(ui, 'TAB_ORG') and ui.TAB_ORG:
                print("✅ Tab organization loaded")
            else:
                print("⚠️  No tab organization found")
                
        except Exception as e:
            print(f"❌ Configuration loading failed: {e}")
            return False
        
        print("✅ Configuration tests completed!")
        return True
        
    except Exception as e:
        print(f"💥 Configuration tests failed: {e}")
        return False

def main():
    """Main test runner function"""
    print_banner()
    
    # Step 1: Prerequisites
    if not check_prerequisites():
        print("\n🛑 Stopping due to prerequisite failures")
        sys.exit(1)
    
    # Step 2: Quick smoke tests
    if not run_quick_smoke_tests():
        print("\n🛑 Stopping due to smoke test failures")
        sys.exit(1)
    
    # Step 3: Configuration tests
    if not run_configuration_tests():
        print("\n⚠️  Configuration issues detected, but continuing...")
    
    # Step 4: Run end-to-end tests
    e2e_results, total_duration = run_all_e2e_tests()
    
    # Step 5: Run additional component tests
    additional_results = run_additional_tests()
    
    # Step 6: Generate report
    all_passed = generate_report(e2e_results, additional_results, total_duration)
    
    # Step 7: Final summary
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED! AgileEE is ready for deployment.")
        print("✅ End-to-end workflows verified")
        print("✅ User scenarios tested")
        print("✅ System integration confirmed")
    else:
        print("⚠️  SOME TESTS FAILED! Please review the results above.")
        print("🔧 Fix failing tests before deployment")
    
    print(f"📊 Test run completed in {total_duration:.2f} seconds")
    print("=" * 80)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)

def run_specific_test_category(category):
    """Run specific category of tests"""
    categories = {
        'workflow': ['tests/test_e2e_complete_workflow.py'],
        'scenarios': ['tests/test_e2e_user_scenarios.py'],
        'integration': ['tests/test_e2e_system_integration.py'],
        'components': [
            'tests/test_model_comparison_tab.py',
            'tests/test_estimator_tab.py', 
            'tests/test_help_tab.py',
            'tests/test_static_shap_tab.py'
        ]
    }
    
    if category not in categories:
        print(f"❌ Unknown category: {category}")
        print(f"Available categories: {', '.join(categories.keys())}")
        return False
    
    print(f"🎯 Running {category} tests...")
    
    results = []
    for test_file in categories[category]:
        if os.path.exists(test_file):
            result = run_test_file(test_file)
            results.append(result)
    
    # Summary for category
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"\n📊 {category.title()} Results: {passed}/{total} passed")
    return passed == total

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "help":
            print("AgileEE E2E Test Runner")
            print("\nUsage:")
            print("  python run_e2e_tests.py           - Run all tests")
            print("  python run_e2e_tests.py smoke     - Run smoke tests only")
            print("  python run_e2e_tests.py workflow  - Run workflow tests")
            print("  python run_e2e_tests.py scenarios - Run user scenario tests")
            print("  python run_e2e_tests.py integration - Run integration tests")
            print("  python run_e2e_tests.py components - Run component tests")
            print("  python run_e2e_tests.py help      - Show this help")
            
        elif command == "smoke":
            print_banner()
            if run_quick_smoke_tests():
                print("✅ Smoke tests passed!")
                sys.exit(0)
            else:
                print("❌ Smoke tests failed!")
                sys.exit(1)
                
        elif command in ["workflow", "scenarios", "integration", "components"]:
            print_banner()
            if check_prerequisites() and run_specific_test_category(command):
                print(f"✅ {command.title()} tests passed!")
                sys.exit(0)
            else:
                print(f"❌ {command.title()} tests failed!")
                sys.exit(1)
        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'python run_e2e_tests.py help' for usage information")
            sys.exit(1)
    else:
        # Run full test suite
        main()