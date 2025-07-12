#!/usr/bin/env python3
"""
CORRECTED AgileEE Test - Matches UI exactly
"""

import sys, os
sys.path.insert(0, 'agileee')
os.environ['STREAMLIT_LOGGER_LEVEL'] = 'ERROR'

import logging
logging.getLogger().setLevel(logging.ERROR)

from agileee.models import predict_man_hours, list_available_models

print("🚀 CORRECTED AgileEE Test - Should Match UI Results")
print("="*70)

# Project 1 - Complete feature set (matching your UI exactly)
project_1_features = {
    'project_prf_year_of_project': 2006,
    'external_eef_industry_sector': 'Banking', 
    'external_eef_organisation_type': 'Banking',
    'project_prf_application_type': 'Financial transaction process/accounting',
    'project_prf_development_type': 'New Development',
    'tech_tf_development_platform': 'MF',
    'tech_tf_language_type': '3GL',
    'tech_tf_primary_programming_language': 'COBOL',
    'project_prf_functional_size': 60,
    'project_prf_relative_size': 'S',
    'project_prf_max_team_size': 8,  # Add reasonable default
    'process_pmf_development_methodologies': 'Agile Development',
    'process_pmf_docs': 3,
    'people_prf_personnel_changes': 0
}

models = list_available_models()
print("Testing Project 1 (COBOL, 60 FP) - Should match your UI results:")
print("Expected: GB=3589, LGBM=2804, BR=5119")
print()

for model in models:
    try:
        pred = predict_man_hours(project_1_features, model['technical_name'])
        print(f"{model['display_name']}: {pred:.0f} hours")
    except Exception as e:
        print(f"{model['display_name']}: ERROR - {e}")

print("\nIf these don't match your UI, the issue is in feature preparation!")