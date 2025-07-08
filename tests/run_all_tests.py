# tests/run_all_tests.py
"""
Updated test runner for UI simplification tests organized in /tests folder.
Provides organized test execution with detailed reporting.
"""

import pytest
import sys
import os
from pathlib import Path
import argparse

# Ensure we're running from the tests directory
TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

class UITestRunner:
    """Organized test runner for UI simplification verification."""
    
    def __init__(self):
        self.tests_dir = TESTS_DIR
        self.results = {}
    
    def run_tab_tests(self, tab_name=None):
        """Run tests for specific tab or all tabs."""
        print(f"\n{'='*60}")
        print(f"🧪 TESTING UI TABS")
        print(f"{'='*60}")
        
        if tab_name:
            test_file = f"tabs/test_{tab_name}_tab.py"
            if not (self.tests_dir / test_file).exists():
                print(f"❌ Test file not found: {test_file}")
                return False
            
            print(f"🎯 Running tests for {tab_name} tab...")
            result = pytest.main([
                str(self.tests_dir / test_file),
                "-v", "--tb=short", "-x"
            ])
            return result == 0
        else:
            # Run all tab tests
            tab_results = {}
            tab_tests = [
                ("estimator", "🔮 Estimator Tab"),
                ("shap", "📊 SHAP Analysis Tab"), 
                ("model_comparison", "🤖 Model Comparison Tab"),
                ("static_shap", "📈 Static SHAP Tab"),
                ("help", "❓ Help Tab")
            ]
            
            for tab_key, tab_display in tab_tests:
                print(f"\n{tab_display}")
                print("-" * 40)
                
                test_file = f"tabs/test_{tab_key}_tab.py"
                if (self.tests_dir / test_file).exists():
                    result = pytest.main([
                        str(self.tests_dir / test_file),
                        "-v", "--tb=short"
                    ])
                    tab_results[tab_key] = result == 0
                else:
                    print(f"⚠️  Test file not found: {test_file}")
                    tab_results[tab_key] = False
            
            self.results.update(tab_results)
            return all(tab_results.values())
    
    def run_unit_tests(self):
        """Run unit tests for individual functions."""
        print(f"\n{'='*60}")
        print(f"🔧 TESTING INDIVIDUAL FUNCTIONS")
        print(f"{'='*60}")
        
        unit_test_dir = self.tests_dir / "unit"
        if not unit_test_dir.exists():
            print("⚠️  Unit test directory not found")
            return False
        
        result = pytest.main([
            str(unit_test_dir),
            "-v", "--tb=short"
        ])
        
        unit_passed = result == 0
        self.results['unit'] = unit_passed
        return unit_passed
    
    def run_integration_tests(self):
        """Run integration tests."""
        print(f"\n{'='*60}")
        print(f"🔗 TESTING INTEGRATION")
        print(f"{'='*60}")
        
        integration_test_dir = self.tests_dir / "integration"
        if not integration_test_dir.exists():
            print("⚠️  Integration test directory not found")
            return False
        
        result = pytest.main([
            str(integration_test_dir),
            "-v", "--tb=short"
        ])
        
        integration_passed = result == 0
        self.results['integration'] = integration_passed
        return integration_passed
    
    def run_removal_verification(self):
        """Run tests to verify configuration management removal."""
        print(f"\n{'='*60}")
        print(f"🗑️  VERIFYING CONFIGURATION REMOVAL")
        print(f"{'='*60}")
        
        removal_tests = [
            "unit/test_removed_functions.py",
            "unit/test_config_removal.py"
        ]
        
        all_passed = True
        for test_file in removal_tests:
            test_path = self.tests_dir / test_file
            if test_path.exists():
                print(f"\n📋 Running {test_file}")
                result = pytest.main([
                    str(test_path),
                    "-v", "--tb=short"
                ])
                if result != 0:
                    all_passed = False
            else:
                print(f"⚠️  Test file not found: {test_file}")
        
        self.results['removal'] = all_passed
        return all_passed
    
    def run_smoke_tests(self):
        """Run smoke tests for core functionality."""
        print(f"\n{'='*60}")
        print(f"💨 RUNNING SMOKE TESTS")
        print(f"{'='*60}")
        
        # Key smoke tests
        smoke_tests = [
            "tabs/test_estimator_tab.py::TestEstimatorTabCore::test_sidebar_inputs_basic_functionality",
            "tabs/test_shap_tab.py::TestShapTabCore::test_shap_tab_with_valid_prediction",
            "tabs/test_model_comparison_tab.py::TestModelComparisonTabCore::test_model_comparison_with_valid_data",
            "unit/test_removed_functions.py::TestConfigFunctionsRemoved::test_make_current_config_json_removed"
        ]
        
        # Filter existing tests
        existing_tests = []
        for test in smoke_tests:
            test_file = test.split("::")[0]
            if (self.tests_dir / test_file).exists():
                existing_tests.append(str(self.tests_dir / test_file))
        
        if not existing_tests:
            print("⚠️  No smoke tests found")
            return False
        
        result = pytest.main([
            *existing_tests,
            "-v", "--tb=line"
        ])
        
        smoke_passed = result == 0
        self.results['smoke'] = smoke_passed
        return smoke_passed
    
    def run_coverage_tests(self):
        """Run tests with coverage reporting."""
        print(f"\n{'='*60}")
        print(f"📊 RUNNING TESTS WITH COVERAGE")
        print(f"{'='*60}")
        
        result = pytest.main([
            str(self.tests_dir),
            "--cov=ui",
            "--cov-report=html:tests/htmlcov",
            "--cov-report=term-missing",
            "--cov-fail-under=70",
            "-v"
        ])
        
        return result == 0
    
    def run_quick_verification(self):
        """Run quick verification of most critical functionality."""
        print(f"\n{'='*60}")
        print(f"⚡ QUICK UI SIMPLIFICATION VERIFICATION")
        print(f"{'='*60}")
        
        # Most critical test patterns
        critical_patterns = [
            "-m smoke",           # Run smoke tests
            "-k 'test_sidebar_inputs_basic_functionality or test_removed'",  # Core + removal
        ]
        
        for pattern in critical_patterns:
            print(f"\n🎯 Running: {pattern}")
            result = pytest.main([
                str(self.tests_dir),
                *pattern.split(),
                "-v", "--tb=line", "-x"
            ])
            
            if result != 0:
                print(f"❌ Quick verification failed at: {pattern}")
                return False
        
        print("✅ Quick verification passed!")
        return True
    
    def run_by_marker(self, marker):
        """Run tests by pytest marker."""
        print(f"\n{'='*60}")
        print(f"🏷️  RUNNING TESTS WITH MARKER: {marker}")
        print(f"{'='*60}")
        
        result = pytest.main([
            str(self.tests_dir),
            f"-m", marker,
            "-v", "--tb=short"
        ])
        
        return result == 0
    
    def print_test_summary(self):
        """Print comprehensive test results summary."""
        print(f"\n{'='*80}")
        print(f"📊 UI SIMPLIFICATION TEST SUMMARY")
        print(f"{'='*80}")
        
        if not self.results:
            print("No test results to display")
            return False
        
        # Define result categories
        categories = [
            ("🔮 Estimator Tab", self.results.get('estimator', None)),
            ("📊 SHAP Analysis Tab", self.results.get('shap', None)),
            ("🤖 Model Comparison Tab", self.results.get('model_comparison', None)),
            ("📈 Static SHAP Tab", self.results.get('static_shap', None)),
            ("❓ Help Tab", self.results.get('help', None)),
            ("🔧 Unit Tests", self.results.get('unit', None)),
            ("🔗 Integration Tests", self.results.get('integration', None)),
            ("🗑️ Config Removal", self.results.get('removal', None)),
            ("💨 Smoke Tests", self.results.get('smoke', None))
        ]
        
        all_passed = True
        for category, result in categories:
            if result is None:
                status = "⚪ SKIP"
            elif result:
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
                all_passed = False
            
            print(f"{category:<25} {status}")
        
        print("="*80)
        
        if all_passed:
            print("🎉 ALL TESTS PASSED! UI SIMPLIFICATION SUCCESSFUL! 🎉")
            print("\n✅ Configuration management successfully removed")
            print("✅ All core functionality preserved")
            print("✅ All tabs work independently")
            print("✅ No save/load dependencies remain")
        else:
            print("⚠️  SOME TESTS FAILED - REVIEW NEEDED")
            print("\n❌ UI simplification may be incomplete")
            print("❌ Check failed tests above for details")
        
        print("="*80)
        return all_passed

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="UI Simplification Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_tests.py --quick              # Quick verification
  python run_all_tests.py --tab estimator      # Test specific tab
  python run_all_tests.py --smoke              # Smoke tests only
  python run_all_tests.py --removed            # Test removed functionality
  python run_all_tests.py --coverage           # Run with coverage
  python run_all_tests.py --marker slow        # Run tests with specific marker
  python run_all_tests.py --all                # Run all tests
        """
    )
    
    # Test selection options
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick verification tests only")
    parser.add_argument("--smoke", action="store_true",
                       help="Run smoke tests only")
    parser.add_argument("--tab", choices=["estimator", "shap", "model_comparison", "static_shap", "help"],
                       help="Run tests for specific tab only")
    parser.add_argument("--unit", action="store_true",
                       help="Run unit tests only")
    parser.add_argument("--integration", action="store_true",
                       help="Run integration tests only")
    parser.add_argument("--removed", action="store_true",
                       help="Run removed functionality verification only")
    parser.add_argument("--coverage", action="store_true",
                       help="Run tests with coverage reporting")
    parser.add_argument("--marker", type=str,
                       help="Run tests with specific pytest marker")
    parser.add_argument("--all", action="store_true",
                       help="Run all tests (default)")
    
    # Output options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Quiet output")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = UITestRunner()
    
    print("🧪 UI SIMPLIFICATION TEST SUITE")
    print(f"📁 Tests directory: {runner.tests_dir}")
    print(f"🎯 Goal: Verify configuration management removal while preserving core functionality")
    
    success = False
    
    try:
        if args.quick:
            success = runner.run_quick_verification()
            
        elif args.smoke:
            success = runner.run_smoke_tests()
            
        elif args.tab:
            success = runner.run_tab_tests(args.tab)
            
        elif args.unit:
            success = runner.run_unit_tests()
            
        elif args.integration:
            success = runner.run_integration_tests()
            
        elif args.removed:
            success = runner.run_removal_verification()
            
        elif args.coverage:
            success = runner.run_coverage_tests()
            
        elif args.marker:
            success = runner.run_by_marker(args.marker)
            
        else:
            # Run all tests (default)
            print("\n🚀 Running comprehensive UI simplification test suite...")
            
            # Run tests in logical order
            tab_success = runner.run_tab_tests()
            unit_success = runner.run_unit_tests()
            integration_success = runner.run_integration_tests()
            removal_success = runner.run_removal_verification()
            smoke_success = runner.run_smoke_tests()
            
            # Print comprehensive summary
            success = runner.print_test_summary()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Test runner error: {e}")
        return 1
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)