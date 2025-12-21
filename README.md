# Cars4You 

## Abstract 
Cars4You is a car resale company seeking to accelerate price evaluations by reducing reliance on manual mechanic inspections. From a machine learning perspective, the task is a supervised regression problem, where the goal is to estimate a continuous target variable (the vehicle resale price) from attributes available at submission time. This project pursued two objectives: (i) develop a regression model to predict car resale prices using information available at submission time and (ii) deliver a lightweight interface that allows employees to obtain predictions either for a single vehicle (form input) or for multiple vehicles (CSV upload).

We began with exploratory data analysis (EDA). Univariate analysis summarized numerical variables through descriptive statistics and distribution visualizations, and assessed categorical variables via counts and category coverage. Multivariate analysis examined relationships between variables, highlighted correlation structure, and revealed inconsistencies and potential redundancy.

Guided by the EDA, we implemented a structured preprocessing pipeline. Operations requiring estimated parameters (e.g., statistical imputation) were implemented under a strict fit/transform design to prevent data leakage during cross-validation. Deterministic cleaning steps (e.g., string standardization, formatting corrections and explicit labeling of missing categories as “unknown”) were applied consistently across training and inference, as they do not depend on validation information. We also engineered domain-informed features (e.g., vehicle age, an ownership flag, log miles per year and engine-size bins). Numerical variables were scaled with StandardScaler and categorical variables were encoded using target encoding for high-cardinality features and one-hot encoding for remaining categories.

We benchmarked several model families (__tree ensembles__, __linear models__, __MLP__, and __ensemble combinations__) using 8-fold cross-validation with random search, prioritizing RMSE and using MAE as a secondary criterion. The best-performing solution was a stacked ensemble, which achieved the strongest overall generalization while remaining compatible with the deployed interface workflow.

## Project Structure 


## Group Member Contributions 
- Francisca Fernandes (20250406): 33%

__Specific Contributions:__ Univariate Analysis, Numerical Variables Preprocessing, 8-Fold Cross Validation Structure, Feature Engineering, Development of RandomForest and HistGradientBoosting Regressors, Prediction Interface 


- Maria Pimentel (20250466): 33%

__Specific Contributions:__ Multivariate Analysis, Scaling, ExtraTrees and Linear Models, Distribution Shift Between Training and Test Data, Model Visualizations, Feature Importances per Price Segment


- Mariana Melo (20250414): 33% 

__Specific Contributions:__ Categorical Data Preprocessing, Feature Selection, Encoding, MLP and Ensemble Models (Bagging, Weighted Average and Stacking)
Specific Contributions: Categorical Data Preprocessing, Feature Selection, Encoding, MLP and Ensemble Models (Bagging, Weighted Average and Stacking)

> Even though these were the main contributions of each member, it is importante to note that every decision was made based on discussions between everyone. When anyone had doubts or uncertainties, we supported each other by reviewing options and solving issues together.

