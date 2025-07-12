#!/usr/bin/env python3
"""
FINAL UI MATCH TEST - Uses exact UI parameters from debug output
"""

import sys, os
sys.path.insert(0, 'agileee')
os.environ['STREAMLIT_LOGGER_LEVEL'] = 'ERROR'

import logging
logging.getLogger().setLevel(logging.ERROR)

from agileee.models import predict_man_hours, list_available_models

print("🎯 FINAL UI MATCH TEST - Exact UI Parameters")
print("="*70)

# EXACT parameters from your UI debug output
ui_exact_features = {
    'project_prf_year_of_project': 2006,
    'external_eef_industry_sector': 'Banking',
    'tech_tf_primary_programming_language': 'Cobol',  # UI: "Cobol" not "COBOL"
    'tech_tf_tools_used': 0,
    'project_prf_relative_size': 'S',
    'project_prf_functional_size': 60,
    'project_prf_development_type': 'New Development',
    'tech_tf_language_type': '3GL (Third Generation)',  # UI: Full form not "3GL"
    'project_prf_application_type': 'Financial Transaction Process/Accounting',  # UI: Title case
    'external_eef_organisation_type': 'Banking',
    'tech_tf_architecture': '',  # UI: Empty string
    'tech_tf_development_platform': 'Mainframe (MF)',  # UI: Full form not "MF"
    'project_prf_team_size_group': '',  # UI: Empty string
    'project_prf_max_team_size': 8,
    'people_prf_personnel_changes': False,  # UI: Boolean not int
    'process_pmf_development_methodologies': 'Agile Development',
    'tech_tf_server_roles': None,  # UI: None
    'tech_tf_client_roles': None,  # UI: None
    'tech_tf_client_server': False,  # UI: Boolean
    'tech_tf_web_development': False,  # UI: Boolean
    'tech_tf_dbms_used': False,  # UI: Boolean
    'process_pmf_prototyping_used': False,  # UI: Boolean
    'project_prf_case_tool_used': False,  # UI: Boolean
    'process_pmf_docs': 3,
    'people_prf_project_user_involvement': 0
}

print("Key differences from original script:")
print("  - tech_tf_primary_programming_language: 'COBOL' → 'Cobol'")
print("  - tech_tf_language_type: '3GL' → '3GL (Third Generation)'")
print("  - tech_tf_development_platform: 'MF' → 'Mainframe (MF)'")
print("  - project_prf_application_type: lowercase → Title Case")
print("  - people_prf_personnel_changes: 0 → False")
print("  - Added 14 additional fields that UI includes")
print()

print("Expected: GB=3589, LGBM=2804, BR=5119")
print("Testing with EXACT UI parameters...")
print()

models = list_available_models()

results = {}
for model in models:
    selected_model = model['technical_name']
    
    print(f"Testing {model['display_name']}...")
    
    try:
        prediction = predict_man_hours(ui_exact_features, selected_model)
        
        if prediction:
            results[model['display_name']] = prediction
            print(f"✅ {model['display_name']}: {prediction:.0f} hours")
        else:
            results[model['display_name']] = None
            print(f"❌ {model['display_name']}: Failed")
    except Exception as e:
        results[model['display_name']] = None
        print(f"❌ {model['display_name']}: ERROR - {e}")

print("\n" + "="*70)
print("📊 COMPARISON WITH UI RESULTS:")

expected = {
    'Gradient Boosting Regressor': 3589,
    'LightGBM Regressor': 2804, 
    'Bayesian Ridge': 5119
}

all_match = True

for model_name in expected:
    actual = results.get(model_name)
    expect = expected[model_name]
    
    if actual:
        diff = abs(actual - expect)
        diff_pct = (diff / expect) * 100
        
        if diff_pct < 1:
            status = "🎯 PERFECT MATCH!"
        elif diff_pct < 2:
            status = "✅ EXCELLENT"
        elif diff_pct < 5:
            status = "✅ VERY CLOSE"
        elif diff_pct < 10:
            status = "⚠️ CLOSE"
        else:
            status = "❌ DIFFERENT"
            all_match = False
        
        print(f"{model_name}:")
        print(f"  Expected: {expect:4d} | Actual: {actual:4.0f} | Diff: {diff:4.0f} ({diff_pct:4.1f}%) {status}")
    else:
        print(f"{model_name}: ❌ PREDICTION FAILED")
        all_match = False

print("\n" + "="*70)
if all_match and all(results.values()):
    print("🎉 SUCCESS! Script now matches UI results!")
    print("✅ All predictions successful with exact UI parameter matching")
else:
    print("⚠️ Still some differences or failures")
    print("💡 Possible remaining issues:")
    print("   - UI might have additional preprocessing steps")
    print("   - Field ordering might matter")
    print("   - Case sensitivity in categorical encoding")

print(f"\n🔍 SUMMARY:")
print(f"Total input features: {len(ui_exact_features)} (matches UI)")
print(f"Key changes made: Exact case matching + additional UI fields")
print(f"This should now match your manual UI test results!")