# constants.py
"""
Central constants file for ML Project Effort Estimator
All magic numbers, file paths, and configuration values in one place
"""

class DataConstants:
    # Column names and keywords
    TARGET_COLUMN = 'project_prf_normalised_work_effort'

        # Size factor mappings
    SIZE_FACTOR_EXTRA_SMALL = 0.5
    SIZE_FACTOR_SMALL = 0.8
    SIZE_FACTOR_MEDIUM = 1.0
    SIZE_FACTOR_LARGE = 1.5
    SIZE_FACTOR_EXTRA_LARGE = 2.0
    
    # Application type factors
    APP_FACTOR_WEB = 1.2
    APP_FACTOR_MOBILE = 1.0
    APP_FACTOR_DESKTOP = 1.3
    APP_FACTOR_API = 0.8
    APP_FACTOR_DEFAULT = 1.0
    
    # Development type factors
    DEV_FACTOR_NEW = 1.0
    DEV_FACTOR_ENHANCEMENT = 0.7
    DEV_FACTOR_MAINTENANCE = 0.5
    DEV_FACTOR_DEFAULT = 1.0
    
    # Estimation constants
    BASE_HOURS_PER_PERSON = 200
    EFFORT_NORMALIZATION_FACTOR = 1000
    
    # Target-related keywords for removal
    TARGET_KEYWORDS = ['target', 'effort', 'label', 'prediction', 'actual', 'ground_truth']
    
    # UI exclusion keys
    UI_EXCLUSION_KEYS = {
        'selected_model', 'selected_models', 'submit', 'clear_results', 
        'show_history', 'save_config', 'config_name', 'comparison_mode'
    }
    
    # ISBSG ignore columns
    ISBSG_IGNORE_COLUMNS = [
        'isbsg_project_id', 'external_eef_data_quality_rating', 
        'external_eef_data_quality_rating_b', 'project_prf_normalised_work_effort_level_1',
        'project_prf_normalised_level_1_pdr_ufp', 'project_prf_normalised_pdr_ufp',
        'project_prf_project_elapsed_time', 'people_prf_ba_team_experience_less_than_1_yr',
        'people_prf_ba_team_experience_1_to_3_yr', 'people_prf_ba_team_experience_great_than_3_yr',
        'people_prf_it_experience_less_than_1_yr', 'people_prf_it_experience_1_to_3_yr',
        'people_prf_it_experience_great_than_3_yr', 'people_prf_it_experience_less_than_3_yr',
        'people_prf_it_experience_3_to_9_yr', 'people_prf_it_experience_great_than_9_yr',
        'people_prf_project_manage_experience', 'project_prf_total_project_cost',
        'project_prf_cost_currency', 'project_prf_currency_multiple', 'project_prf_speed_of_delivery',
        'people_prf_project_manage_changes', 'project_prf_defect_density', 'project_prf_manpower_delivery_rate'
    ]
    
    # Mixed type columns for preprocessing
    MIXED_TYPE_COLUMNS = [
        'external_eef_industry_sector',
        'tech_tf_client_roles', 
        'tech_tf_clientserver_description',
        'tech_tf_development_platform_hand_held'
    ]

    
class FeatureValidationConstants:
    """Configuration-driven feature validation using YAML"""
    
    @classmethod
    def get_valid_industries(cls):
        """Get valid industries from feature_mapping.yaml"""
        from agileee.config_loader import ConfigLoader
        import os
        
        config_path = os.path.join(FileConstants.CONFIG_FOLDER, FileConstants.FEATURE_MAPPING_FILE)
        config = ConfigLoader.load_yaml_config(config_path)
        if config:
            return config.get('categorical_features', {}).get('external_eef_industry_sector', {}).get('options', [])
        return []
    
    @classmethod
    def get_valid_languages(cls):
        """Get valid programming languages from feature_mapping.yaml"""
        from agileee.config_loader import ConfigLoader
        import os
        
        config_path = os.path.join(FileConstants.CONFIG_FOLDER, FileConstants.FEATURE_MAPPING_FILE)
        config = ConfigLoader.load_yaml_config(config_path)
        if config:
            return config.get('categorical_features', {}).get('tech_tf_primary_programming_language', {}).get('options', [])
        return []
    
    @classmethod
    def get_mandatory_fields(cls):
        """Get mandatory fields from ui_info.yaml for pipeline cols_to_keep"""
        from agileee.config_loader import ConfigLoader
        import os
        
        config_path = os.path.join(FileConstants.CONFIG_FOLDER, FileConstants.UI_INFO_FILE)
        config = ConfigLoader.load_yaml_config(config_path)
        if config:
            fields = config.get('fields', {})
            return [field_name for field_name, field_config in fields.items() 
                   if field_config.get('mandatory', False)]
        return []

class FileConstants:
    # Folder paths
    CONFIG_FOLDER = 'config'
    MODELS_FOLDER = 'models'
    DATA_FOLDER = 'data'
    
    # Configuration files
    UI_INFO_FILE = 'ui_info.yaml'
    FEATURE_MAPPING_FILE = 'feature_mapping.yaml'
    MODEL_DISPLAY_NAME_FILE = 'model_display_names.json'
    SHAP_ANALYSIS_FILE = 'shap_analysis.md'
    DYNAMIC_FULL_MODEL_FEATURES_FILE = 'full_model_features.json'
    
    # Pipeline files
    PIPELINE_FILE = 'preprocessing_pipeline.pkl'
    PIPELINE_MODEL_FILE = 'finance_pycaret_preprocessing_pipeline.pkl'
    
    # Data files
    ISBSG_PREPROCESSED_FILE = 'finance_shap_background_random200.csv'


class LoggingConstants:
    # Log file names
    APP_LOG_FILE = "app.log"
    
    # Log format
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Log levels by component
    DEFAULT_LOG_LEVEL = "INFO"
    SHAP_LOG_LEVEL = "WARNING"

class ModelConstants:
    # Model file extensions
    MODEL_EXTENSION = '.pkl'
    JOBLIB_EXTENSION = '.joblib'
    # Target optimization candidates
    TARGET_OPTIMIZATION_CANDIDATES = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


    
    # Prediction validation ranges
    REASONABLE_PREDICTION_MIN_OPTIMIZED = 10
    REASONABLE_PREDICTION_MAX_OPTIMIZED = 5000
    
    # Expected feature counts
    EXPECTED_FEATURE_COUNT_PIPELINE = 67

    # Model types for SHAP compatibility
    TREE_MODEL_KEYWORDS = [
        'forest', 'tree', 'xgb', 'lgb', 'catboost', 'gradient',
        'randomforest', 'extratrees', 'decisiontree'
    ]
    
    LINEAR_MODEL_KEYWORDS = [
        'linear', 'lasso', 'ridge', 'elastic', 'bayesianridge'
    ]
    
    # Model display name mappings
    MODEL_TYPE_MAPPINGS = {
        'rf': 'Random Forest',
        'xgb': 'XGBoost', 
        'lgb': 'LightGBM',
        'lr': 'Linear Regression',
        'svm': 'Support Vector Machine',
        'dt': 'Decision Tree',
        'nb': 'Naive Bayes',
        'knn': 'K-Nearest Neighbors',
        'ada': 'AdaBoost',
        'gb': 'Gradient Boosting',
        'et': 'Extra Trees'
    }

class PipelineConstants:
    # Pipeline parameters
    HIGH_MISSING_THRESHOLD = 0.7
    MAX_CATEGORICAL_CARDINALITY = 10
    DEFAULT_SAMPLE_SIZE = 100
    KERNEL_EXPLAINER_SAMPLE_SIZE = 50
    
    # Processing limits
    MAX_FEATURES_SHOWN = 15
    PRECISION_DECIMALS = 3
    TOP_N_FEATURES = 10
    MAX_ANALYSIS_POINTS = 20

    # Team size grouping thresholds
    TEAM_SIZE_SINGLE_THRESHOLD = 1
    TEAM_SIZE_SMALL_THRESHOLD = 3
    TEAM_SIZE_MEDIUM_THRESHOLD = 5
    TEAM_SIZE_LARGE_THRESHOLD = 10
    
    # Team size group labels
    TEAM_SIZE_SINGLE_LABEL = "1"
    TEAM_SIZE_SMALL_LABEL = "2-3"
    TEAM_SIZE_MEDIUM_LABEL = "4-5"
    TEAM_SIZE_LARGE_LABEL = "6-10"
    TEAM_SIZE_XLARGE_LABEL = "11+"
    
    # Cardinality and threshold constants
    DEFAULT_MAX_CARDINALITY = 10
    MISSING_VALUE_WARNING_THRESHOLD = 0.5
    PREDICTION_MAX_CARDINALITY = 50
    PREDICTION_MISSING_THRESHOLD = 0.95

    DEFAULT_COLS_TO_KEEP = [
            'project_prf_case_tool_used', 
            'process_pmf_prototyping_used',
            'tech_tf_client_roles', 
            'tech_tf_type_of_server', 
            'tech_tf_clientserver_description'
        ]

class ShapConstants:
    # SHAP Model type detection keywords
    TREE_MODEL_KEYWORDS = [
        'forest', 'tree', 'xgb', 'lgb', 'catboost', 'gradient',
        'randomforest', 'extratrees', 'decisiontree'
    ]
    
    LINEAR_MODEL_KEYWORDS = [
        'linear', 'lasso', 'ridge', 'elastic', 'bayesianridge'
    ]


class ValidationConstants:
    # Validation thresholds
    MIN_PREDICTION_VALUE = 0.1
    MAX_PREDICTION_VALUE = 1000000
    
    # Feature validation
    MIN_FEATURE_COUNT = 5
    MAX_FEATURE_COUNT = 200
    
    # Data quality thresholds
    MAX_MISSING_RATIO = 0.95
    MAX_INFINITE_VALUES = 0
    MEMORY_WARNING_MB = 100

    REASONABLE_PREDICTION_MIN = 10
    REASONABLE_PREDICTION_MAX = 5000
    EXPECTED_FEATURE_COUNT_MIN = 50
    EXPECTED_FEATURE_COUNT_MAX = 70

class UIConstants:
    # UI markers and symbols
    REQUIRED_FIELD_MARKER = "⭐"
    
    # Default values
    DEFAULT_TEAM_SIZE = 5
    DEFAULT_FUNCTIONAL_SIZE = 100
    HOURS_PER_DAY = 8
    DAYS_PER_WEEK = 5
    WEEKS_PER_MONTH = 4.33
    
    # Prediction thresholds
    LOW_PREDICTION_WARNING = 10
    HIGH_PREDICTION_WARNING = 192000
    HIGH_VARIANCE_THRESHOLD = 0.3  # 30%
    
    # SHAP analysis
    SHAP_CACHE_SIZE_WARNING = 50
    WHAT_IF_MIN_POINTS = 5
    WHAT_IF_MAX_POINTS = 20
    WHAT_IF_DEFAULT_POINTS = 10

    # SHAP display constants
    SHAP_FEATURE_PREVIEW_COUNT = 10
    SHAP_CHART_DEFAULT_HEIGHT = 500
    SHAP_FEATURE_DISPLAY_LIMIT = 5

    # Display formatting
    COLUMN_SPLIT_RATIO = 2  # For mid = len(items) // COLUMN_SPLIT_RATIO
    PREDICTION_DECIMAL_PLACES = 0
    IMPORTANCE_DISPLAY_PRECISION = 3
    
    # Metric display templates
    EFFORT_METRIC_TEMPLATE = "📊 Total Effort"
    DAYS_METRIC_TEMPLATE = "📅 Working Days"
    WEEKS_METRIC_TEMPLATE = "📆 Working Weeks"
    MONTHS_METRIC_TEMPLATE = "🗓️ Months"







