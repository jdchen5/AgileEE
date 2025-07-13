#!/usr/bin/env python3
"""
Simplified Model Test Script - Uses external test data file
"""

import sys, os
sys.path.insert(0, 'agileee')
os.environ['STREAMLIT_LOGGER_LEVEL'] = 'ERROR'

import logging
logging.getLogger().setLevel(logging.ERROR)

import pandas as pd
from agileee.models import predict_man_hours, list_available_models

# Import test data
try:
    from test_data import TEST_DATA
    print(f"✅ Loaded {len(TEST_DATA)} test cases from test_data.py")
except ImportError:
    print("❌ Could not import test_data.py")
    print("Please create test_data.py with your TEST_DATA list")
    sys.exit(1)

def convert_to_ui_format(data_row):
    """Convert training data format to UI format with proper mappings"""
    
    # Map development platform abbreviations
    platform_mapping = {
        'MF': 'Mainframe (MF)',
        'PC': 'PC', 
        'Multi': 'Multi-platform'
    }
    
    # Map language type abbreviations  
    language_type_mapping = {
        '3GL': '3GL (Third Generation)',
        '4GL': '4GL (Fourth Generation)'
    }
    
    # Map programming languages to UI format
    language_mapping = {
        'COBOL': 'Cobol',
        'PL/I': 'PL/i',
        'Visual Basic': 'Other',
        'Oracle': 'Oracle',
        'Java': 'JAVA',
        'iOS': 'Other',
        '.Net': 'Other'
    }
    
    # Map application types
    app_type_mapping = {
        'Financial transaction process/accounting': 'Financial Transaction Process/Accounting',
        'Business Application': 'Business Application',
        'Mobile Application': 'Mobile Application'
    }
    
    # Map architecture values
    architecture_mapping = {
        'Stand alone': 'Stand-alone',
        'Client server': 'Client-Server'
    }
    
    # Create UI format
    ui_row = {
        'project_prf_year_of_project': data_row.get('project_prf_year_of_project', 2024),
        'external_eef_industry_sector': data_row.get('external_eef_industry_sector', ''),
        'tech_tf_primary_programming_language': data_row.get('tech_tf_primary_programming_language', ''),
        'tech_tf_tools_used': data_row.get('tech_tf_tools_used') or 0,
        'project_prf_relative_size': data_row.get('project_prf_relative_size', ''),
        'project_prf_functional_size': data_row.get('project_prf_functional_size', 100),
        'project_prf_development_type': data_row.get('project_prf_development_type', ''),
        'tech_tf_language_type': data_row.get('tech_tf_language_type', ''),
        'project_prf_application_type': data_row.get('project_prf_application_type', ''),
        'external_eef_organisation_type': data_row.get('external_eef_organisation_type', ''),
        'tech_tf_architecture': data_row.get('tech_tf_architecture', ''),
        'tech_tf_development_platform': data_row.get('tech_tf_development_platform', ''),
        'project_prf_team_size_group': data_row.get('project_prf_team_size_group', ''),
        'project_prf_max_team_size': data_row.get('project_prf_max_team_size') or 10,
        'people_prf_personnel_changes': False,
        'process_pmf_development_methodologies': data_row.get('process_pmf_development_methodologies', ''),
        'tech_tf_server_roles': data_row.get('tech_tf_server_roles', ''),
        'tech_tf_client_roles': data_row.get('tech_tf_client_roles', ''),
        'tech_tf_client_server': False,
        'tech_tf_web_development': False,
        'tech_tf_dbms_used': False,
        'process_pmf_prototyping_used': False,
        'project_prf_case_tool_used': False,
        'process_pmf_docs': data_row.get('process_pmf_docs') or 0,
        'people_prf_project_user_involvement': data_row.get('people_prf_project_user_involvement') or 0
    }
    
    # Apply mappings
    if ui_row['tech_tf_development_platform'] in platform_mapping:
        ui_row['tech_tf_development_platform'] = platform_mapping[ui_row['tech_tf_development_platform']]
    
    if ui_row['tech_tf_language_type'] in language_type_mapping:
        ui_row['tech_tf_language_type'] = language_type_mapping[ui_row['tech_tf_language_type']]
    
    if ui_row['tech_tf_primary_programming_language'] in language_mapping:
        ui_row['tech_tf_primary_programming_language'] = language_mapping[ui_row['tech_tf_primary_programming_language']]
    
    if ui_row['project_prf_application_type'] in app_type_mapping:
        ui_row['project_prf_application_type'] = app_type_mapping[ui_row['project_prf_application_type']]
    
    if ui_row['tech_tf_architecture'] in architecture_mapping:
        ui_row['tech_tf_architecture'] = architecture_mapping[ui_row['tech_tf_architecture']]
    
    # Handle Yes/No boolean fields
    yes_no_fields = ['tech_tf_client_server', 'tech_tf_web_development', 'tech_tf_dbms_used', 
                     'process_pmf_prototyping_used', 'project_prf_case_tool_used']
    
    for field in yes_no_fields:
        if data_row.get(field) == 'Yes':
            ui_row[field] = True
        elif data_row.get(field) == 'No':
            ui_row[field] = False
    
    # Handle web development special case
    if data_row.get('tech_tf_web_development') == 'Web':
        ui_row['tech_tf_web_development'] = True
    
    # Convert None values to appropriate defaults
    for key, value in ui_row.items():
        if value is None:
            if key in ['project_prf_max_team_size', 'project_prf_functional_size', 
                      'tech_tf_tools_used', 'process_pmf_docs', 'people_prf_project_user_involvement']:
                ui_row[key] = 0
            elif key.startswith('tech_tf_') and key.endswith(('_server', '_development', '_used')):
                ui_row[key] = False
            else:
                ui_row[key] = ''
    
    return ui_row

def main():
    print("🎯 COMPREHENSIVE MODEL TEST")
    print("="*80)
    
    # Get available models
    try:
        models = list_available_models()
        print(f"Found {len(models)} models: {[m['display_name'] for m in models]}")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Prepare results storage
    results = []
    
    # Run predictions for each test case
    for i, test_case in enumerate(TEST_DATA, 1):
        print(f"\n📋 Test Case {i:2d}: {test_case['project_prf_year_of_project']} | "
              f"{test_case['external_eef_industry_sector']} | "
              f"Size: {test_case['project_prf_relative_size']} | "
              f"FP: {test_case['project_prf_functional_size']} | "
              f"Actual: {test_case['project_prf_normalised_work_effort']} hours")
        
        # Convert to UI format
        ui_features = convert_to_ui_format(test_case)
        
        # Store test case info
        test_result = {
            'Test_Case': i,
            'Year': test_case['project_prf_year_of_project'],
            'Industry': test_case['external_eef_industry_sector'],
            'Size': test_case['project_prf_relative_size'],
            'Functional_Points': test_case['project_prf_functional_size'],
            'Max_Team_Size': ui_features['project_prf_max_team_size'],
            'Actual_Hours': test_case['project_prf_normalised_work_effort'],
            'Language': test_case.get('tech_tf_primary_programming_language', 'N/A'),
            'Platform': test_case.get('tech_tf_development_platform', 'N/A'),
            'Dev_Type': test_case.get('project_prf_development_type', 'N/A')
        }
        
        # Test each model
        for model in models:
            model_name = model['display_name']
            technical_name = model['technical_name']
            
            try:
                prediction = predict_man_hours(ui_features, technical_name)
                if prediction:
                    test_result[f'{model_name}_Hours'] = int(prediction)
                    actual = test_case['project_prf_normalised_work_effort']
                    error_pct = abs(prediction - actual) / actual * 100
                    test_result[f'{model_name}_Error_%'] = round(error_pct, 1)
                    print(f"    {model_name:25s}: {prediction:6.0f} hours (Error: {error_pct:5.1f}%)")
                else:
                    test_result[f'{model_name}_Hours'] = 'FAILED'
                    test_result[f'{model_name}_Error_%'] = 'N/A'
                    print(f"    {model_name:25s}: FAILED")
            except Exception as e:
                test_result[f'{model_name}_Hours'] = 'ERROR'
                test_result[f'{model_name}_Error_%'] = 'N/A'
                print(f"    {model_name:25s}: ERROR - {str(e)[:50]}")
        
        results.append(test_result)
    
    # Create and display results table
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE RESULTS TABLE")
    print("="*80)
    
    df = pd.DataFrame(results)
    
    # Display summary table
    print("\n🎯 PREDICTION SUMMARY:")
    summary_cols = ['Test_Case', 'Year', 'Industry', 'Size', 'Functional_Points', 'Max_Team_Size', 'Actual_Hours']
    for model in models:
        summary_cols.append(f"{model['display_name']}_Hours")
    
    print(df[summary_cols].to_string(index=False, max_colwidth=15))
    
    # Display error analysis
    print("\n📈 ERROR ANALYSIS (% difference from actual):")
    error_cols = ['Test_Case', 'Size', 'Actual_Hours']
    for model in models:
        error_cols.append(f"{model['display_name']}_Error_%")
    
    print(df[error_cols].to_string(index=False, max_colwidth=15))
    
    # Calculate model statistics
    print("\n📊 MODEL PERFORMANCE STATISTICS:")
    print("-" * 60)
    
    for model in models:
        model_name = model['display_name']
        hours_col = f'{model_name}_Hours'
        error_col = f'{model_name}_Error_%'
        
        # Filter out failed predictions
        valid_predictions = df[df[hours_col] != 'FAILED'][df[hours_col] != 'ERROR']
        
        if len(valid_predictions) > 0:
            errors = pd.to_numeric(valid_predictions[error_col], errors='coerce')
            valid_errors = errors.dropna()
            
            if len(valid_errors) > 0:
                print(f"\n{model_name}:")
                print(f"  Successful Predictions: {len(valid_predictions)}/{len(df)}")
                print(f"  Mean Absolute Error:    {valid_errors.mean():.1f}%")
                print(f"  Median Error:           {valid_errors.median():.1f}%")
                print(f"  Min Error:              {valid_errors.min():.1f}%")
                print(f"  Max Error:              {valid_errors.max():.1f}%")
                
                # Performance categories
                excellent = (valid_errors <= 10).sum()
                good = ((valid_errors > 10) & (valid_errors <= 25)).sum()
                poor = (valid_errors > 25).sum()
                
                print(f"  Excellent (≤10%):       {excellent} cases")
                print(f"  Good (10-25%):          {good} cases")
                print(f"  Poor (>25%):            {poor} cases")
        else:
            print(f"\n{model_name}: No successful predictions")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETED")
    print(f"📈 Tested {len(TEST_DATA)} cases across {len(models)} models")
    print("="*80)

if __name__ == "__main__":
    main()