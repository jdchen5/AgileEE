import pandas as pd
import numpy as np

def create_agileee_sample_dataframe():
    """
    Create a DataFrame with all the sample data from your AgileEE application
    with proper field names matching your UI
    """
    
    # Sample data rows based on your provided data
    data = [
        {
            'project_prf_year_of_project': 2006,
            'external_eef_industry_sector': 'Banking',
            'external_eef_organisation_type': 'Banking',
            'project_prf_application_group': 'Business Application',
            'project_prf_application_type': 'Financial transaction process/accounting',
            'project_prf_development_type': 'New Development',
            'tech_tf_development_platform': 'MF',
            'tech_tf_language_type': '3GL',
            'tech_tf_primary_programming_language': 'COBOL',
            'project_prf_functional_size': 60,
            'functional_size_group': 'S:<100',
            'project_prf_relative_size': 'S',
            'project_prf_normalised_work_effort': 3730,
            'project_prf_team_size_group': None,
            'project_prf_max_team_size': None,
            'project_prf_case_tool_used': None,
            'process_pmf_development_methodologies': 'Agile Development',
            'process_pmf_prototyping_used': None,
            'process_pmf_docs': 3,
            'tech_tf_architecture': None,
            'tech_tf_client_server': None,
            'tech_tf_client_roles': None,
            'tech_tf_server_roles': None,
            'tech_tf_type_of_server': None,
            'tech_tf_clientserver_description': None,
            'tech_tf_web_development': None,
            'tech_tf_dbms_used': None,
            'tech_tf_tools_used': None,
            'people_prf_project_user_involvement': None,
            'people_prf_project_manage_changes': None,
            'people_prf_personnel_changes': 0
        },
        {
            'project_prf_year_of_project': 2009,
            'external_eef_industry_sector': 'Financial',
            'external_eef_organisation_type': 'Financial, Property & Business Services',
            'project_prf_application_group': 'Business Application',
            'project_prf_application_type': 'Financial transaction process/accounting',
            'project_prf_development_type': 'Enhancement',
            'tech_tf_development_platform': 'MF',
            'tech_tf_language_type': '3GL',
            'tech_tf_primary_programming_language': 'PL/I',
            'project_prf_functional_size': 90,
            'functional_size_group': 'S:<100',
            'project_prf_relative_size': 'S',
            'project_prf_normalised_work_effort': 5831,
            'project_prf_team_size_group': '5-8',
            'project_prf_max_team_size': 8,
            'project_prf_case_tool_used': 'No',
            'process_pmf_development_methodologies': 'Agile Development',
            'process_pmf_prototyping_used': None,
            'process_pmf_docs': 13,
            'tech_tf_architecture': 'Stand alone',
            'tech_tf_client_server': 'No',
            'tech_tf_client_roles': None,
            'tech_tf_server_roles': None,
            'tech_tf_type_of_server': None,
            'tech_tf_clientserver_description': None,
            'tech_tf_web_development': 'Yes',
            'tech_tf_dbms_used': None,
            'tech_tf_tools_used': 2,
            'people_prf_project_user_involvement': None,
            'people_prf_project_manage_changes': None,
            'people_prf_personnel_changes': 0
        },
        {
            'project_prf_year_of_project': 2012,
            'external_eef_industry_sector': 'Banking',
            'external_eef_organisation_type': 'Banking',
            'project_prf_application_group': 'Business Application',
            'project_prf_application_type': 'Business Application',
            'project_prf_development_type': 'New Development',
            'tech_tf_development_platform': None,
            'tech_tf_language_type': '4GL',
            'tech_tf_primary_programming_language': None,
            'project_prf_functional_size': 63,
            'functional_size_group': 'S:<100',
            'project_prf_relative_size': 'S',
            'project_prf_normalised_work_effort': 1784,
            'project_prf_team_size_group': None,
            'project_prf_max_team_size': None,
            'project_prf_case_tool_used': None,
            'process_pmf_development_methodologies': 'Agile Development',
            'process_pmf_prototyping_used': None,
            'process_pmf_docs': 2,
            'tech_tf_architecture': None,
            'tech_tf_client_server': None,
            'tech_tf_client_roles': None,
            'tech_tf_server_roles': None,
            'tech_tf_type_of_server': None,
            'tech_tf_clientserver_description': None,
            'tech_tf_web_development': None,
            'tech_tf_dbms_used': None,
            'tech_tf_tools_used': None,
            'people_prf_project_user_involvement': None,
            'people_prf_project_manage_changes': None,
            'people_prf_personnel_changes': 0
        },
        {
            'project_prf_year_of_project': 2006,
            'external_eef_industry_sector': 'Banking',
            'external_eef_organisation_type': 'Banking',
            'project_prf_application_group': 'Business Application',
            'project_prf_application_type': 'Financial transaction process/accounting',
            'project_prf_development_type': 'Enhancement',
            'tech_tf_development_platform': 'MF',
            'tech_tf_language_type': '3GL',
            'tech_tf_primary_programming_language': 'COBOL',
            'project_prf_functional_size': 143,
            'functional_size_group': 'M:100-999',
            'project_prf_relative_size': 'M1',
            'project_prf_normalised_work_effort': 9566,
            'project_prf_team_size_group': None,
            'project_prf_max_team_size': None,
            'project_prf_case_tool_used': None,
            'process_pmf_development_methodologies': 'Agile Development',
            'process_pmf_prototyping_used': None,
            'process_pmf_docs': 3,
            'tech_tf_architecture': None,
            'tech_tf_client_server': None,
            'tech_tf_client_roles': None,
            'tech_tf_server_roles': None,
            'tech_tf_type_of_server': None,
            'tech_tf_clientserver_description': None,
            'tech_tf_web_development': None,
            'tech_tf_dbms_used': None,
            'tech_tf_tools_used': None,
            'people_prf_project_user_involvement': None,
            'people_prf_project_manage_changes': None,
            'people_prf_personnel_changes': 0
        },
        {
            'project_prf_year_of_project': 2007,
            'external_eef_industry_sector': 'Banking',
            'external_eef_organisation_type': 'Banking',
            'project_prf_application_group': 'Business Application',
            'project_prf_application_type': 'Financial transaction process/accounting',
            'project_prf_development_type': 'Enhancement',
            'tech_tf_development_platform': 'PC',
            'tech_tf_language_type': '3GL',
            'tech_tf_primary_programming_language': 'Visual Basic',
            'project_prf_functional_size': 577,
            'functional_size_group': 'M:100-999',
            'project_prf_relative_size': 'M2',
            'project_prf_normalised_work_effort': 9382,
            'project_prf_team_size_group': None,
            'project_prf_max_team_size': None,
            'project_prf_case_tool_used': None,
            'process_pmf_development_methodologies': 'Agile Development',
            'process_pmf_prototyping_used': None,
            'process_pmf_docs': 3,
            'tech_tf_architecture': None,
            'tech_tf_client_server': None,
            'tech_tf_client_roles': None,
            'tech_tf_server_roles': None,
            'tech_tf_type_of_server': None,
            'tech_tf_clientserver_description': None,
            'tech_tf_web_development': None,
            'tech_tf_dbms_used': None,
            'tech_tf_tools_used': None,
            'people_prf_project_user_involvement': None,
            'people_prf_project_manage_changes': None,
            'people_prf_personnel_changes': 0
        },
        {
            'project_prf_year_of_project': 2009,
            'external_eef_industry_sector': 'Banking',
            'external_eef_organisation_type': 'Banking',
            'project_prf_application_group': 'Business Application',
            'project_prf_application_type': 'Financial transaction process/accounting',
            'project_prf_development_type': 'Enhancement',
            'tech_tf_development_platform': 'MF',
            'tech_tf_language_type': '3GL',
            'tech_tf_primary_programming_language': 'COBOL',
            'project_prf_functional_size': 271,
            'functional_size_group': 'M:100-999',
            'project_prf_relative_size': 'M1',
            'project_prf_normalised_work_effort': 2613,
            'project_prf_team_size_group': None,
            'project_prf_max_team_size': None,
            'project_prf_case_tool_used': None,
            'process_pmf_development_methodologies': 'Agile Development',
            'process_pmf_prototyping_used': None,
            'process_pmf_docs': 3,
            'tech_tf_architecture': None,
            'tech_tf_client_server': None,
            'tech_tf_client_roles': None,
            'tech_tf_server_roles': None,
            'tech_tf_type_of_server': None,
            'tech_tf_clientserver_description': None,
            'tech_tf_web_development': None,
            'tech_tf_dbms_used': None,
            'tech_tf_tools_used': None,
            'people_prf_project_user_involvement': None,
            'people_prf_project_manage_changes': None,
            'people_prf_personnel_changes': 0
        },
        {
            'project_prf_year_of_project': 2011,
            'external_eef_industry_sector': 'Banking',
            'external_eef_organisation_type': 'Banking',
            'project_prf_application_group': 'Business Application',
            'project_prf_application_type': 'Business Application',
            'project_prf_development_type': 'New Development',
            'tech_tf_development_platform': None,
            'tech_tf_language_type': '4GL',
            'tech_tf_primary_programming_language': 'Visual Basic',
            'project_prf_functional_size': 160,
            'functional_size_group': 'M:100-999',
            'project_prf_relative_size': 'M1',
            'project_prf_normalised_work_effort': 989,
            'project_prf_team_size_group': None,
            'project_prf_max_team_size': None,
            'project_prf_case_tool_used': None,
            'process_pmf_development_methodologies': 'Agile Development',
            'process_pmf_prototyping_used': None,
            'process_pmf_docs': 2,
            'tech_tf_architecture': None,
            'tech_tf_client_server': None,
            'tech_tf_client_roles': None,
            'tech_tf_server_roles': None,
            'tech_tf_type_of_server': None,
            'tech_tf_clientserver_description': None,
            'tech_tf_web_development': None,
            'tech_tf_dbms_used': None,
            'tech_tf_tools_used': None,
            'people_prf_project_user_involvement': None,
            'people_prf_project_manage_changes': None,
            'people_prf_personnel_changes': 0
        },
        {
            'project_prf_year_of_project': 2013,
            'external_eef_industry_sector': 'Financial',
            'external_eef_organisation_type': 'Financial, Property & Business Services',
            'project_prf_application_group': 'Business Application',
            'project_prf_application_type': 'Business Application',
            'project_prf_development_type': 'New Development',
            'tech_tf_development_platform': None,
            'tech_tf_language_type': '4GL',
            'tech_tf_primary_programming_language': 'Oracle',
            'project_prf_functional_size': 336,
            'functional_size_group': 'M:100-999',
            'project_prf_relative_size': 'M2',
            'project_prf_normalised_work_effort': 3494,
            'project_prf_team_size_group': None,
            'project_prf_max_team_size': None,
            'project_prf_case_tool_used': None,
            'process_pmf_development_methodologies': 'Agile Development',
            'process_pmf_prototyping_used': None,
            'process_pmf_docs': 2,
            'tech_tf_architecture': None,
            'tech_tf_client_server': None,
            'tech_tf_client_roles': None,
            'tech_tf_server_roles': None,
            'tech_tf_type_of_server': None,
            'tech_tf_clientserver_description': None,
            'tech_tf_web_development': None,
            'tech_tf_dbms_used': None,
            'tech_tf_tools_used': None,
            'people_prf_project_user_involvement': None,
            'people_prf_project_manage_changes': None,
            'people_prf_personnel_changes': 0
        },
        {
            'project_prf_year_of_project': 2014,
            'external_eef_industry_sector': 'Banking',
            'external_eef_organisation_type': 'Banking',
            'project_prf_application_group': 'Business Application',
            'project_prf_application_type': 'Business Application',
            'project_prf_development_type': 'New Development',
            'tech_tf_development_platform': None,
            'tech_tf_language_type': '3GL',
            'tech_tf_primary_programming_language': 'Java',
            'project_prf_functional_size': 958,
            'functional_size_group': 'M:100-999',
            'project_prf_relative_size': 'M2',
            'project_prf_normalised_work_effort': 13200,
            'project_prf_team_size_group': None,
            'project_prf_max_team_size': None,
            'project_prf_case_tool_used': None,
            'process_pmf_development_methodologies': 'Agile Development',
            'process_pmf_prototyping_used': None,
            'process_pmf_docs': 2,
            'tech_tf_architecture': None,
            'tech_tf_client_server': None,
            'tech_tf_client_roles': None,
            'tech_tf_server_roles': None,
            'tech_tf_type_of_server': None,
            'tech_tf_clientserver_description': None,
            'tech_tf_web_development': None,
            'tech_tf_dbms_used': None,
            'tech_tf_tools_used': None,
            'people_prf_project_user_involvement': None,
            'people_prf_project_manage_changes': None,
            'people_prf_personnel_changes': 0
        },
        {
            'project_prf_year_of_project': 2006,
            'external_eef_industry_sector': 'Banking',
            'external_eef_organisation_type': 'Banking',
            'project_prf_application_group': 'Business Application',
            'project_prf_application_type': 'Financial transaction process/accounting',
            'project_prf_development_type': 'New Development',
            'tech_tf_development_platform': 'PC',
            'tech_tf_language_type': '3GL',
            'tech_tf_primary_programming_language': 'Java',
            'project_prf_functional_size': 1006,
            'functional_size_group': 'L:1000-2999',
            'project_prf_relative_size': 'L',
            'project_prf_normalised_work_effort': 23429,
            'project_prf_team_size_group': None,
            'project_prf_max_team_size': None,
            'project_prf_case_tool_used': None,
            'process_pmf_development_methodologies': 'Agile Development',
            'process_pmf_prototyping_used': None,
            'process_pmf_docs': 3,
            'tech_tf_architecture': None,
            'tech_tf_client_server': None,
            'tech_tf_client_roles': None,
            'tech_tf_server_roles': None,
            'tech_tf_type_of_server': None,
            'tech_tf_clientserver_description': None,
            'tech_tf_web_development': 'Web',
            'tech_tf_dbms_used': None,
            'tech_tf_tools_used': None,
            'people_prf_project_user_involvement': None,
            'people_prf_project_manage_changes': None,
            'people_prf_personnel_changes': 0
        },
        {
            'project_prf_year_of_project': 2013,
            'external_eef_industry_sector': 'Banking',
            'external_eef_organisation_type': 'Banking',
            'project_prf_application_group': 'Business Application',
            'project_prf_application_type': 'Mobile Application',
            'project_prf_development_type': 'New Development',
            'tech_tf_development_platform': 'PC',
            'tech_tf_language_type': '3GL',
            'tech_tf_primary_programming_language': 'iOS',
            'project_prf_functional_size': 1244,
            'functional_size_group': 'L:1000-2999',
            'project_prf_relative_size': 'L',
            'project_prf_normalised_work_effort': 5773,
            'project_prf_team_size_group': '5-8',
            'project_prf_max_team_size': 8,
            'project_prf_case_tool_used': 'No',
            'process_pmf_development_methodologies': 'Agile Development',
            'process_pmf_prototyping_used': None,
            'process_pmf_docs': 3,
            'tech_tf_architecture': 'Client server',
            'tech_tf_client_server': 'Yes',
            'tech_tf_client_roles': 'mobile app',
            'tech_tf_server_roles': 'back-end',
            'tech_tf_type_of_server': 'back-end',
            'tech_tf_clientserver_description': None,
            'tech_tf_web_development': None,
            'tech_tf_dbms_used': None,
            'tech_tf_tools_used': None,
            'people_prf_project_user_involvement': None,
            'people_prf_project_manage_changes': None,
            'people_prf_personnel_changes': 0
        },
        {
            'project_prf_year_of_project': 2009,
            'external_eef_industry_sector': 'Banking',
            'external_eef_organisation_type': 'Banking',
            'project_prf_application_group': 'Business Application',
            'project_prf_application_type': 'Financial transaction process/accounting',
            'project_prf_development_type': 'Enhancement',
            'tech_tf_development_platform': 'MF',
            'tech_tf_language_type': '3GL',
            'tech_tf_primary_programming_language': 'COBOL',
            'project_prf_functional_size': 1252,
            'functional_size_group': 'L:1000-2999',
            'project_prf_relative_size': 'L',
            'project_prf_normalised_work_effort': 14437,
            'project_prf_team_size_group': None,
            'project_prf_max_team_size': None,
            'project_prf_case_tool_used': None,
            'process_pmf_development_methodologies': 'Agile Development',
            'process_pmf_prototyping_used': None,
            'process_pmf_docs': 3,
            'tech_tf_architecture': None,
            'tech_tf_client_server': None,
            'tech_tf_client_roles': None,
            'tech_tf_server_roles': None,
            'tech_tf_type_of_server': None,
            'tech_tf_clientserver_description': None,
            'tech_tf_web_development': None,
            'tech_tf_dbms_used': None,
            'tech_tf_tools_used': None,
            'people_prf_project_user_involvement': None,
            'people_prf_project_manage_changes': None,
            'people_prf_personnel_changes': 0
        },
        {
            'project_prf_year_of_project': 2010,
            'external_eef_industry_sector': 'Banking',
            'external_eef_organisation_type': 'Banking',
            'project_prf_application_group': None,
            'project_prf_application_type': None,
            'project_prf_development_type': 'New Development',
            'tech_tf_development_platform': 'Multi',
            'tech_tf_language_type': '4GL',
            'tech_tf_primary_programming_language': '.Net',
            'project_prf_functional_size': 5667,
            'functional_size_group': 'XL:3000-8999',
            'project_prf_relative_size': 'XL',
            'project_prf_normalised_work_effort': 50457,
            'project_prf_team_size_group': None,
            'project_prf_max_team_size': None,
            'project_prf_case_tool_used': None,
            'process_pmf_development_methodologies': 'Agile Development',
            'process_pmf_prototyping_used': None,
            'process_pmf_docs': 2,
            'tech_tf_architecture': 'Client server',
            'tech_tf_client_server': 'Yes',
            'tech_tf_client_roles': None,
            'tech_tf_server_roles': None,
            'tech_tf_type_of_server': None,
            'tech_tf_clientserver_description': None,
            'tech_tf_web_development': None,
            'tech_tf_dbms_used': None,
            'tech_tf_tools_used': None,
            'people_prf_project_user_involvement': None,
            'people_prf_project_manage_changes': None,
            'people_prf_personnel_changes': 0
        }
    ]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

def analyze_dataframe(df):
    """
    Analyze the AgileEE sample dataframe
    """
    print("=== AgileEE Sample Data Analysis ===")
    print(f"DataFrame Shape: {df.shape}")
    print(f"Total Records: {len(df)}")
    print(f"🎯 TARGET COLUMN: project_prf_normalised_work_effort")
    print()
    
    # TARGET ANALYSIS FIRST
    target_col = 'project_prf_normalised_work_effort'
    if target_col in df.columns:
        target_values = df[target_col].dropna()
        print("=== TARGET VARIABLE ANALYSIS ===")
        print(f"Target: {target_col}")
        print(f"  Records: {len(target_values)}")
        print(f"  Min Effort: {target_values.min():,.0f} hours")
        print(f"  Max Effort: {target_values.max():,.0f} hours")
        print(f"  Mean Effort: {target_values.mean():,.0f} hours")
        print(f"  Median Effort: {target_values.median():,.0f} hours")
        print(f"  Std Dev: {target_values.std():,.0f} hours")
        print()
    
    print("=== FEATURE FIELD SUMMARY ===")
    for col in df.columns:
        if col != target_col:  # Exclude target from feature analysis
            non_null_count = df[col].notna().sum()
            null_count = df[col].isna().sum()
            unique_values = df[col].nunique()
            print(f"{col}: {non_null_count} non-null, {null_count} null, {unique_values} unique")
    
    print("\n=== Categorical Field Value Counts ===")
    categorical_fields = [
        'external_eef_industry_sector',
        'external_eef_organisation_type', 
        'project_prf_application_type',
        'project_prf_development_type',
        'tech_tf_development_platform',
        'tech_tf_language_type',
        'tech_tf_primary_programming_language',
        'project_prf_relative_size'
    ]
    
    for field in categorical_fields:
        if field in df.columns:
            print(f"\n{field}:")
            value_counts = df[field].value_counts()
            for value, count in value_counts.items():
                print(f"  {value}: {count}")
    
    print("\n=== Numeric Field Statistics ===")
    numeric_fields = [
        'project_prf_year_of_project',
        'project_prf_functional_size',
        'project_prf_normalised_work_effort',
        'project_prf_max_team_size'
    ]
    
    for field in numeric_fields:
        if field in df.columns:
            series = df[field].dropna()
            if len(series) > 0:
                print(f"\n{field}:")
                print(f"  Min: {series.min()}")
                print(f"  Max: {series.max()}")
                print(f"  Mean: {series.mean():.2f}")
                print(f"  Median: {series.median():.2f}")

def save_to_csv(df, filename="agileee_sample_data.csv"):
    """
    Save the dataframe to CSV
    """
    df.to_csv(filename, index=False)
    print(f"DataFrame saved to {filename}")

def get_features_and_target(df):
    """
    Split the dataframe into features (X) and target (y) for ML purposes
    
    Returns:
        X_features: DataFrame with all feature columns
        y_target: Series with target values
        feature_names: List of feature column names
    """
    target_col = 'project_prf_normalised_work_effort'
    
    # Separate features from target
    feature_columns = [col for col in df.columns if col != target_col]
    
    X_features = df[feature_columns].copy()
    y_target = df[target_col].copy()
    
    print(f"✅ Features (X): {X_features.shape[1]} columns, {X_features.shape[0]} rows")
    print(f"🎯 Target (y): {len(y_target)} values")
    print(f"📋 Feature columns: {feature_columns[:5]}... (showing first 5)")
    
    return X_features, y_target, feature_columns

def get_test_records_for_prediction():
    """
    Get test records formatted for PREDICTION TESTING (features only, no target)
    """
    df = create_agileee_sample_dataframe()
    X_features, y_target, feature_names = get_features_and_target(df)
    
    # Select representative records for testing
    test_cases = {
        'banking_small_project': {
            'description': 'Small COBOL banking project (actual effort: 3,730 hours)',
            'expected_effort_range': (3000, 4500),
            'features': X_features.iloc[0].to_dict()
        },
        'banking_medium_project': {
            'description': 'Medium Visual Basic banking project (actual effort: 9,382 hours)', 
            'expected_effort_range': (8000, 11000),
            'features': X_features.iloc[4].to_dict()
        },
        'banking_large_project': {
            'description': 'Large Java banking project (actual effort: 23,429 hours)',
            'expected_effort_range': (20000, 27000), 
            'features': X_features.iloc[9].to_dict()
        },
        'mobile_app_project': {
            'description': 'iOS mobile banking app (actual effort: 5,773 hours)',
            'expected_effort_range': (5000, 7000),
            'features': X_features.iloc[10].to_dict()
        },
        'large_dotnet_project': {
            'description': 'Large .NET enterprise project (actual effort: 50,457 hours)',
            'expected_effort_range': (45000, 55000),
            'features': X_features.iloc[12].to_dict()
        }
    }
    
    return test_cases

def create_training_test_split(df, test_size=0.2, random_state=42):
    """
    Create train/test split for model evaluation
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_features, y_target, feature_names = get_features_and_target(df)
    
    # Simple train/test split (you can use sklearn for more sophisticated splitting)
    n_test = int(len(df) * test_size)
    test_indices = list(range(0, n_test))
    train_indices = list(range(n_test, len(df)))
    
    X_train = X_features.iloc[train_indices]
    X_test = X_features.iloc[test_indices] 
    y_train = y_target.iloc[train_indices]
    y_test = y_target.iloc[test_indices]
    
    print(f"📊 Training set: {X_train.shape}")
    print(f"🧪 Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Create the dataframe
    df = create_agileee_sample_dataframe()
    
    # Display basic info
    print("AgileEE Sample Data DataFrame Created")
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows (excluding target for privacy):")
    feature_cols = [col for col in df.columns if col != 'project_prf_normalised_work_effort']
    print(df[feature_cols[:5]].head())  # Show first 5 feature columns only
    
    # Analyze the data
    print("\n")
    analyze_dataframe(df)
    
    # Show features vs target separation
    print("\n")
    X_features, y_target, feature_names = get_features_and_target(df)
    
    # Show target statistics with project context
    print("\n=== TARGET VARIABLE BY PROJECT SIZE ===")
    size_effort = df.groupby('project_prf_relative_size')['project_prf_normalised_work_effort'].agg(['count', 'mean', 'min', 'max'])
    for size, stats in size_effort.iterrows():
        print(f"{size}: {stats['count']} projects, avg {stats['mean']:,.0f} hours (range: {stats['min']:,.0f} - {stats['max']:,.0f})")
    
    # Save to CSV
    save_to_csv(df)
    
    # Get test cases for PREDICTION testing (features only)
    test_cases = get_test_records_for_prediction()
    print(f"\n=== TEST CASES FOR PREDICTION TESTING ===")
    for case_name, case_info in test_cases.items():
        print(f"\n{case_name.upper()}:")
        print(f"  Description: {case_info['description']}")
        print(f"  Expected Range: {case_info['expected_effort_range'][0]:,} - {case_info['expected_effort_range'][1]:,} hours")
        print(f"  Key Features:")
        features = case_info['features']
        for field in ['project_prf_functional_size', 'tech_tf_primary_programming_language', 'project_prf_development_type'][:3]:
            if field in features and features[field] is not None:
                print(f"    {field}: {features[field]}")
    
    # Create train/test split example
    print(f"\n=== TRAIN/TEST SPLIT EXAMPLE ===")
    X_train, X_test, y_train, y_test = create_training_test_split(df)
    
    print(f"\n🎯 Ready for ML model training and testing!")
    print(f"   - Use X_train, y_train for training your models")
    print(f"   - Use X_test, y_test for evaluation") 
    print(f"   - Use test_cases for UI prediction testing")