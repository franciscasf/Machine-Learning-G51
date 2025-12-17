# Cars 4 You — Expediting Car Evaluations with ML

## Project Overview
**Cars 4 You** is an online car resale company aiming to reduce operational bottlenecks in vehicle valuation by replacing manual mechanical inspections with a fast, data-driven price 
estimation system.  
The objective of this project is to design, evaluate, and deploy a **robust regression pipeline** capable of predicting car purchase prices using only user-provided attributes available 
at registration time.

The project follows a full ML lifecycle: exploratory analysis, preprocessing, feature engineering and selection, regression benchmarking, model optimization, open-ended investigation, 
and final deployment via Kaggle submission.

---

## Methodological Framework
### Cross-Validation 
To obtain a reliable estimate of out-of-sample performance and to compare models under the same conditions, every model we developed was evaluated using the  same cross-validation 
framework. This avoids overfitting to a single train/validation split and ensures that performance comparisons between models are fair and reproducible.

We use an **8-fold** cross-validation scheme:
- The data is split into `N_SPLITS = 8` approximately equal folds.
- At each iteration, 1 fold is used as the **validation set** and the remaining 7 folds are used as the **training set**.
- `shuffle = True` is used to randomize the split, and a fixed `RANDOM_STATE = 42` guarantees reproducibility of the folds.
- This configuration is applied consistently across all developed models, so differences in performance reflect the models themselves rather than differences in the data split.

For each model and configuration, we record per-fold metrics for both **train** and **validation** sets and later aggregate them to summarize performance.

### Shared evaluation metrics

All models are evaluated using the same regression metrics:

- **RMSE (Root Mean Squared Error)**  
  Penalizes larger errors more strongly; used as the primary measure of overall prediction error.  
  `RMSE = sqrt( (1/n) * Σ (y_i − ŷ_i)² )`

- **MAE (Mean Absolute Error)**  
  Measures the average absolute deviation between predictions and true values; more robust to outliers than RMSE and the metric used for Kaggle ranking.  
  `MAE = (1/n) * Σ |y_i − ŷ_i|`

- **R² (Coefficient of Determination)**  
  Proportion of variance in the target explained by the model.  
  `R² = 1 − Σ (y_i − ŷ_i)² / Σ (y_i − ȳ)²`

- **Bias (Signed Error)**  
  Captures systematic overprediction (positive) or underprediction (negative).  
  `Bias = (1/n) * Σ (ŷ_i − y_i)`

For each fold, metrics are computed on both **train** and **validation** sets.  
Comparing train vs validation values allows diagnosis of underfitting and overfitting, while mean validation metrics across folds provide a consistent basis for model selection.

---

## Models Evaluated
Each model was introduced with a clear purpose and evaluated under the same preprocessing and validation logic.

### Linear Models (Baseline & Regularised)
- **Linear Regression (OLS)**: baseline for interpretability and error scale reference.
- **Ridge / Lasso / ElasticNet**: explored bias-variance trade-offs and feature sparsity.
- **Key insight**: linear assumptions were insufficient to capture the dominant non-linearities and interactions in the data, and some predictions were simply not reasonsable, with
  negative prices being predicted.

**Performance:** competitive baseline, but systematically outperformed by non-linear models.

### Tree-Based Ensembles
- **Random Forest Regressor**: reduced variance and captured non-linear effects.
- **Extra Trees Regressor**: aggressive randomisation, low bias, higher overfitting risk.
- **HistGradientBoosting Regressor (HGB)**: strongest single model, combining efficiency, stability, and accuracy.

**Performance:** HGB achieved the best validation and Kaggle scores, with controlled overfitting and consistent generalisation.

### Neural Networks
- **Feedforward Neural Network (MLP)**: trained on engineered numerical and encoded categorical features.
- Designed to test whether flexible, fully non-linear function approximation could outperform tree-based ensembles.
- Required careful tuning of architecture depth, regularisation, and early stopping to maintain stability.

**Performance:** competitive but less robust than gradient-boosted trees, with higher sensitivity to hyperparameters and preprocessing choices.

---

## Model Comparison Summary
| Model Family | Purpose | Outcome |
|---|---|---|
| Linear Models | Baseline & interpretability | Underfit |
| Regularised Linear | Feature control | Limited gains |
| Random Forest | Non-linear baseline | Strong |
| Extra Trees | Low-bias exploration | Overfit |
| HistGradientBoosting | Final selected model | **Best** |

---

## Open-Ended Section: Additional Insights Beyond Benchmarking

The open-ended component was designed to complement pure leaderboard optimisation by evaluating **robustness, interpretability, and practical usability** of our solution.
We organised this section into five aligned investigations.

### 1) Model Combination Strategies (Ensembling)
To test whether different learners capture complementary error patterns, we evaluated three ensemble approaches, each trained and evaluated using leakage-safe out-of-fold procedures.

- **Bagging Regressor (Decision Trees):** bootstrap aggregation of multiple **DecisionTreeRegressor** models to reduce variance relative to a single tree.  
- **Weighted Average (HGB + RF):** optimised weighted mean of predictions from **HistGradientBoostingRegressor** and **RandomForestRegressor**, aiming to combine boosting's low-bias
  behaviour with RF's variance reduction.  
- **Stacking Regressor (HGB + RF + ET → Ridge meta-model):** base learners were **HistGradientBoostingRegressor**, **RandomForestRegressor**, and **ExtraTreesRegressor**, with **Ridge
  regression** as the meta-learner trained on out-of-fold predictions.

**Takeaway:** ensembling provided incremental improvements and increased robustness, particularly in harder-to-predict regions, although gains were naturally bounded by the strength of the best single estimator.

**BEST FINAL MODEL:** stacking ??????????

---

### 2) Distribution Shift Between Training and Test Data
We assessed whether the Kaggle test set exhibited meaningful **covariate shift** relative to training (i.e., differences in feature distributions that could compromise generalisation). The observed shift was **minimal**, suggesting that performance differences between offline validation and Kaggle are unlikely to be driven by dataset mismatch.

**Takeaway:** our validation protocol is representative of the test environment, and no additional shift-correction steps were warranted.

---

### 3) Feature Importance Across Target Segments (Tier-Dependent Mechanisms)
Beyond global feature importance, we analysed how feature relevance changes across **different price tiers**. This segment-based analysis indicates that, even under a single global model, the learned pricing mechanism is **tier-dependent**:

- **Low-price cars:** dominated by depreciation/usage effects (`age`, `mileage`) plus baseline model effects.
- **High-price cars:** dominated by product identity and specifications (`model`, `engineSize`, `mpg`, `Brand`), with depreciation still relevant but no longer the primary driver.

**Takeaway:** global importance can hide heterogeneity in the decision mechanism; results motivate future extensions such as **tier-specific calibration** or **specialised models per price segment**.

---

### 4) Interactive Analytics Interface 
To connect modeling with practical usage, we implemented an interactive interface where users can:
- enter car attributes manually (single prediction), or
- upload a CSV with multiple cars (batch prediction),

and receive predicted prices using the final deployed pipeline.

**Takeaway:** this demonstrates deployment-readiness and ensures the model can be used as an operational decision-support tool rather than only as a Kaggle artifact.

---

### 5) Ablation Study: Measuring the Value of Each Pipeline Block
To quantify the contribution of each design choice, we ran a controlled ablation study across models using five standardised variants:  
1. **Baseline:** all original features, no feature engineering (FE) / feature selection (FS), raw price.
2. **Simplified + Log target:** drop `previousOwners`, predict `log1p(price)`.
3. **+ Depreciation proxy:** variant 2 plus replace `year` with `age`.
4. **+ Feature selection:** variant 3 plus FS keeping 80% of features.
5. **Full pipeline:** variant 4 plus all engineered features, FS keeping 65%.

Each variant was evaluated under the same cross-validation framework, enabling direct attribution of performance changes to specific pipeline steps.


#### Insights from Elastic Net Variants: 
Since our primary goal is to minimise **RMSE**, the Elastic Net ablation results indicate that the largest RMSE reductions come from transforming the target rather than adding
complexity.  
- The **raw-price baseline (v1)** has the worst validation RMSE by a large margin, showing that linear modeling on the original target scale struggles with price skew and large-error
  cases.
- Switching to a **log-transformed target** and simplifying features (**v2**) produces a **major RMSE drop**, indicating that stabilising the target distribution is **the most impactful
  intervention for a linear model**.
- Replacing `year` with an explicit depreciation proxy `age` (**v3**) delivers a small but consistent additional RMSE improvement over v2.
- Adding feature selection (**v4**, FS80) slightly worsens RMSE relative to v2-v3, suggesting that for this model the removed features still contributed to reducing large errors.
- The most complex configuration (**v5**, full feature engineering with stronger feature selection) increases RMSE further, consistent with diminishing returns and potential information
  loss from aggressive selection.

**Conclusion (Elastic Net):** to reduce RMSE in linear models, the key is **log-target training + `age`**, while stronger FS / heavy feature engineering is not RMSE-optimal.

#### Insights from Extra Trees Variants:
For Extra Trees, RMSE behaviour differs because the model already captures non-linearities and interactions without requiring extensive feature engineering.  
- **The original-feature Extra Trees model (v1) achieves the lowest validation RMSE** among the tested variants, showing that **the model performs best when given the full raw feature
  set**. 
- Moving to a **log-target** setup with slight changes in features (**v2-v3**) increases RMSE slightly, indicating that these adjustments do not improve large-error behaviour for Extra
  Trees in our setting.
- Introducing feature selection (**v4**, FS80) does not reduce RMSE and remains worse than v1, consistent with some loss of useful split information.
- The most complex configuration (**v5**, full feature engineering + FS65) produces the **worst RMSE** among the Extra Trees variants, suggesting that additional engineered predictors
  increased noise or removed relevant signal for large-error cases.

**Conclusion (Extra Trees):** for RMSE minimisation, Extra Trees performed best with the **simplest configuration (all original features, no FE/FS, raw target)**; increasing pipeline 
complexity did not reduce RMSE and generally hurt performance.


**Takeaway:** the ablation study clarifies which steps consistently improve generalisation (and which add complexity without measurable benefit), supporting a transparent justification 
of the final design.

---

## Deployment
The final pipeline was retrained on the full training dataset using the selected configuration and applied to the test set to generate predictions for Kaggle submission.  
Deployment reproduces the same preprocessing and modeling steps used during cross-validation, ensuring consistency between offline evaluation and online performance.

---

## Conclusions
- Non-linear ensemble methods substantially outperform linear baselines for car price prediction.
- Careful control of preprocessing and validation is as impactful as model choice.
- Apparent outliers often encode real economic/mechanical signal, and naive correction can degrade performance.
- HistGradientBoosting provided the best balance between accuracy, robustness, and practical deployment.

Overall, this project shows that methodological discipline and systematic experimentation are more valuable than model complexity alone when building reliable predictive systems.
