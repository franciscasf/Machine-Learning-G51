# Cars4You - Expediting Car Evaluations with ML

This project was developed in the context of the **Machine Learning** course of the **MSc in Data Science and Advanced Analytics at NOVA IMS**. It was carried out as a group project and focused on applying machine learning methods to a realistic business problem involving used-car price prediction, from data analysis and preprocessing to model selection and deployment. 

**Project grade: 19/20.**

## Abstract 
Cars4You is a car resale company seeking to accelerate price evaluations by reducing reliance on manual mechanic inspections. From a machine learning perspective, the task is a supervised regression problem, where the goal is to estimate a continuous target variable (the vehicle resale price) from attributes available at submission time. This project pursued two objectives: (i) develop a regression model to predict car resale prices using information available at submission time and (ii) deliver a lightweight interface that allows employees to obtain predictions either for a single vehicle (form input) or for multiple vehicles (CSV upload).

We began with exploratory data analysis (EDA). Univariate analysis summarized numerical variables through descriptive statistics and distribution visualizations, and assessed categorical variables via counts and category coverage. Multivariate analysis examined relationships between variables, highlighted correlation structure, and revealed inconsistencies and potential redundancy.

Guided by the EDA, we implemented a structured preprocessing pipeline. Operations requiring estimated parameters (e.g., statistical imputation) were implemented under a strict fit/transform design to prevent data leakage during cross-validation. Deterministic cleaning steps (e.g., string standardization, formatting corrections and explicit labeling of missing categories as “unknown”) were applied consistently across training and inference, as they do not depend on validation information. We also engineered domain-informed features (e.g., vehicle age, an ownership flag, log miles per year and engine-size bins). Numerical variables were scaled with StandardScaler and categorical variables were encoded using target encoding for high-cardinality features and one-hot encoding for remaining categories.

We benchmarked several model families (__tree ensembles__, __linear models__, __MLP__, and __ensemble combinations__) using 8-fold cross-validation with random search, prioritizing RMSE and using MAE as a secondary criterion. The best-performing solution was a stacked ensemble, which achieved the strongest overall generalization while remaining compatible with the deployed interface workflow.

## Project Structure 
<pre style="white-space: pre; font-family: monospace; margin: 0;">
README.md
</pre>

<pre style="white-space: pre; font-family: monospace; margin: 0;">
README2.md
</pre>

<pre style="white-space: pre; font-family: monospace; margin: 0;">
notebooks/
├── 00_preproc_helpers.ipynb
├── 01_EDA.ipynb
├── 02_visualization_helpers.ipynb
├── 03_MethodologicalFramework.ipynb
├── 04_LinearModels.ipynb
├── 05_RandomForest.ipynb
├── 06_ExtraTrees.ipynb
├── 07_HistGradientBoosting.ipynb
├── 08_NeuralNetworks.ipynb
├── 09_Weighted_Mean.ipynb
├── 10_BaggingRegressor.ipynb
├── 11_Stacking.ipynb
├── 12_Stacking_elasticnet.ipynb
├── 13_ModelComparisons.ipynb
├── 14_Open_Ended.ipynb
└── final_submission_stacking.csv
</pre>

<pre style="white-space: pre; font-family: monospace; margin: 0;">
interface/
├── README.md
├── app.py
├── backend/
│   ├── preproc_helper.py
│   └── stack_best_config.py
└── pages/
    └── 1_Predict.py
</pre>


## Group Member Contributions 
- Francisca Fernandes (20250406): 33%

__Specific Contributions:__ Univariate Analysis, Numerical Variables Preprocessing, 8-Fold Cross Validation Structure, Feature Engineering, Development of RandomForest and HistGradientBoosting Regressors, Prediction Interface 


- Maria Pimentel (20250466): 33%

__Specific Contributions:__ Multivariate Analysis, Scaling, ExtraTrees and Linear Models, Distribution Shift Between Training and Test Data, Model Visualizations, Feature Importances per Price Segment


- Mariana Melo (20250414): 33% 

__Specific Contributions:__ Categorical Data Preprocessing, Feature Selection, Encoding, MLP and Ensemble Models (Bagging, Weighted Average and Stacking)

> Even though these were the main contributions of each member, it is importante to note that every decision was made based on discussions between everyone. When anyone had doubts or uncertainties, we supported each other by reviewing options and solving issues together.


## References

Géron, A. (2017). Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems (1st ed.). O'Reilly Media.

scikit-learn developers. (n.d.). Common pitfalls and recommended practices. scikit-learn documentation. From https://scikit-learn.org/stable/common_pitfalls

scikit-learn developers. (n.d.). sklearn.ensemble.RandomForestRegressor. scikit-learn documentation. From https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

scikit-learn developers. (n.d.). sklearn.ensemble.HistGradientBoostingRegressor. scikit-learn documentation. From https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html#sklearn.ensemble.HistGradientBoostingRegressor

GOV.UK. (n.d.). Vehicle tax rates: Vehicle tax rate tables. Retrieved December 1, 2025, from
https://www.gov.uk/vehicle-tax-rate-tables
https://www.gov.uk/vehicle-tax-rate-tables/rates-for-cars-registered-on-or-after-1-march-2001

