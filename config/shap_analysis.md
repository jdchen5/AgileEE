# SHAP Analysis Report — Feature Importance Explained

---

## Model 1: Light Gradient Boosting Machine Regressor

![Light Gradient Boosting Machine Regressor SHAP Summary](../plots/shap_summary_LGBMRegressor.png)

### What this shows  
This SHAP summary plot ranks features by their overall importance in the LightGBM Regressor model’s predictions. Each dot represents a project instance, colored by feature value (red = high, blue = low). The x-axis (SHAP value) shows the impact and direction of the feature on the model’s output.

### Why these features matter  
- **`project_prf_functional_size`** has the highest impact, indicating that larger functional size strongly increases predicted effort.  
- Technology-related features such as **`tech_tf_language_type_3gl`** and team size features , **`project_prf_max_team_size`** are also influential, reflecting the role of technical complexity and staffing.

### Interpretation  
High values of important features (shown by red dots on the right) typically lead to higher predicted project effort, while low values (blue) are associated with lower effort predictions. This helps users understand the main drivers behind the LightGBM model’s estimates.

---

## Model 2: Gradient Boosting Regressor

![Gradient Boosting Regressor SHAP Summary](../plots/shap_summary_GradientBoostingRegressor.png)

### What this shows  
This SHAP summary plot visualizes which features most influence the predictions of the Gradient Boosting Regressor model. Each dot represents a project example, colored by feature value (red = high, blue = low). The x-axis position (SHAP value) indicates the magnitude and direction of the feature’s impact on predicted project effort.

### Why these features matter  
- **`project_prf_functional_size`** is the top feature, so the overall size of the project has the greatest influence on estimated effort.  
- **Team and technology factors** like project_prf_max_team_size, tech_tf_language_type_3gl, and tech_tf_tools_used are also highly impactful, showing that both people and technical stack contribute significantly to predictions.
- **Contextual features** such as external_eef_industry_sector_banking and external_eef_organisation_type_transport_and_storage suggest that the type of industry or organization can shift the expected project effort.

### Interpretation  
Projects with higher values in key features—like larger functional size, bigger teams, and advanced technology types—tend to have higher predicted effort (red dots on the right). Conversely, lower values (blue) are linked with lower effort. This plot helps identify what project characteristics drive the Gradient Boosting model’s predictions, making the model’s reasoning more transparent for stakeholders.

---

## Model 3: LassoLars Regressor

![LassoLars SHAP Summary](../plots/shap_summary_LassoLars)

### What this shows  
This SHAP summary plot displays the most influential features in the predictions made by the Lasso Least Angle Regression (LassoLars) Regressor. Each point is a project example, with color representing the value of the feature (red = high, blue = low). The x-axis shows how much each feature pushes the prediction higher or lower (the SHAP value).

### Why these features matter  
- project_prf_functional_size is the dominant driver, indicating that projects with greater functional size have much higher estimated effort.
- Team size and technical features like project_prf_max_team_size, tech_tf_language_type_3gl, and tech_tf_tools_used remain strong predictors, highlighting the impact of team structure and technology choices.
- Organizational and contextual variables such as external_eef_industry_sector_banking and external_eef_organisation_type_transport_and_storage show that business environment and sector can meaningfully affect the model’s output.

### Interpretation  
Larger functional size, bigger teams, and advanced technologies (high feature values, red dots on the right) lead to higher predicted project effort. Conversely, projects with smaller scope, smaller teams, or simpler tech stacks (blue dots) are linked to lower effort estimates. This visualization makes it clear which project characteristics are most influential in the LassoLars model’s decision-making.

---

## Summary

- **SHAP values provide clear insights** into which features affect the model predictions and in what way.  
- **`project_prf_functional_size` is consistently the top driver** across models, confirming its critical role in effort estimation.  
- Technology and project characteristics also play significant roles.  
- This transparency helps users understand and trust the machine learning models.

---

## How to Use This Report

- Review feature importance plots to see which factors most influence effort predictions.  
- Use the explanations to communicate to stakeholders why certain projects are predicted to require more or less effort.  
- Combine this understanding with domain knowledge to make better project planning decisions.

---

*End of SHAP Analysis Report*
