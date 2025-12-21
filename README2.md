# Cars 4 You - Expediting Car Evaluations with ML

## Project Overview
**Cars 4 You** is an online car resale company aiming to reduce operational bottlenecks in vehicle valuation by replacing manual mechanical inspections with a fast, data-driven price 
estimation system.  
The objective of this project is to design, evaluate, and deploy a **robust regression pipeline** capable of predicting car purchase prices using only user-provided attributes available at registration time.

The project follows a full ML lifecycle: exploratory analysis, preprocessing, feature engineering and selection, regression benchmarking, model optimization, open-ended investigation, 
and final deployment via Kaggle submission.

---

## Methodological Framework
### Cross-Validation 
To obtain a reliable estimate of out-of-sample performance and to compare models under the same conditions, every model we developed was evaluated using the  same cross-validation 
framework. This avoids overfitting to a single train/validation split and ensures that performance comparisons between models are fair and reproducible.

- The data is split into `N_SPLITS = 8` approximately equal folds.
- At each iteration, 1 fold is used as the **validation set** and the remaining 7 folds are used as the **training set**.
- `shuffle = True` is used to randomize the split, and a fixed `RANDOM_STATE = 42` guarantees reproducibility of the folds.
This configuration is applied consistently across all developed models, so differences in performance reflect the models themselves rather than differences in the data split.

For each model and configuration, we record per-fold metrics for both **train** and **validation** sets and later aggregate them to summarize performance.

### Shared evaluation metrics

All models are analyzed using the same regression metrics:

- **RMSE (Root Mean Squared Error):** penalizes larger errors more strongly; we used it as the **primary measure of overall prediction error**.
- **MAE (Mean Absolute Error):** measures the average absolute deviation between predictions and true values; more robust to outliers than RMSE, and the metric used for Kaggle ranking; for multiple models with similar RMSE values, MAE served as a *tie-breaker*.
- **R²:** proportion of variance in the target explained by the model.  
- **Bias (Signed Error):** captures systematic overprediction (positive) or underprediction (negative).
  
For each fold, metrics are computed on both **train** and **validation** sets.  
Comparing train vs validation values allows diagnosis of underfitting and overfitting, while mean validation metrics across folds provide a consistent basis for model selection.

---

## Models Evaluated
Each model was introduced with a clear purpose and evaluated under the same preprocessing and validation logic.

### Linear Models (Baseline & Regularised)
- **Linear Regression (OLS)**: baseline for interpretability and error scale reference.
- **Ridge / Lasso / ElasticNet**: explored bias-variance trade-offs and feature sparsity.
  
**Key insight**: linear assumptions did not capture the dominant non-linearities and interactions in the data (not even feature engineering, that added some non-linear relations, was enough to improve the model's accuracy enough to make it competitive), and some predictions were simply **not reasonable**, with negative prices being predicted.

**Performance:** ElasticNet performed slightly better than the others, but they were all systematically outperformed by non-linear models, as expected.

### Tree-Based Ensembles
- **Random Forest Regressor**: reduced variance and captured non-linear effects; big overfit.
- **Extra Trees Regressor**: aggressive randomisation, low bias, higher overfitting.
- **HistGradientBoosting Regressor (HGB)**: strongest single model, combining efficiency, stability, and accuracy.

**Performance:** HGB achieved the best validation and Kaggle scores, with controlled overfitting and consistent generalisation.

### Neural Networks
- **Feedforward Neural Network (MLP)**: designed to test whether flexible, fully non-linear function approximation could outperform tree-based ensembles.

**Performance:** less robust than gradient-boosted trees, with higher sensitivity to hyperparameters and preprocessing choices.

---

## Model Comparison Summary
| Model Family             | Purpose                     | Outcome                |
|--------------------------|-----------------------------|------------------------|
| OLS                      | Baseline & interpretability | Underfit               |
| Regularised Linear       | Feature control             | Limited gains          |
| Random Forest            | Non-linear baseline         | Strong but overfitting |
| Extra Trees              | Low-bias exploration        | Higher overfit         |
| **HistGradientBoosting** | **Final selected model**    | **Best**               |
| MLP                      | Fully non-linear function   | Less robust            |

---

## Open-Ended Section: Additional Insights Beyond Benchmarking

The open-ended component was designed to complement pure leaderboard optimisation by evaluating **robustness, interpretability, and practical usability** of our solution.
We organised this section into five aligned investigations.

### 1) Model Combination Strategies (Ensembling)
To test whether different learners capture complementary error patterns, we evaluated three ensemble approaches:  
- **Bagging Regressor (Decision Trees):** bootstrap aggregation of multiple DecisionTreeRegressor models to reduce variance relative to a single tree.  
- **Weighted Average (HGB + RF):** optimised weighted mean of predictions from HistGradientBoostingRegressor and RandomForestRegressor, aiming to combine boosting's low-bias
  behaviour with RF's variance reduction.  
- **Stacking Regressor (HGB + RF + ET → Ridge meta-model):** base learners were HistGradientBoostingRegressor, RandomForestRegressor, and ExtraTreesRegressor, with Ridge
  regression as the meta-learner trained on out-of-fold predictions.

Ensembling provided incremental improvements and increased robustness, particularly in harder-to-predict regions, although gains were naturally bounded by the strength of the best single estimator.  

---

### 2) Distribution Shift Between Training and Test Data
We assessed whether the Kaggle test set exhibited meaningful **covariate shift** relative to training (differences in feature distributions that could compromise generalisation). The observed shift was **minimal**, suggesting that performance differences between offline validation and Kaggle are unlikely to be driven by dataset mismatch.

---

### 3) Feature Importance Across Target Segments (Tier-Dependent Mechanisms)
Beyond global feature importance, we analysed how feature relevance changes across **different price tiers**. This segment-based analysis indicates that, even under a single global model, the learned pricing mechanism is **tier-dependent**:  
- **Low-price cars:** dominated by depreciation/usage effects (`age`, `mileage`) plus baseline `model` effects.
- **High-price cars:** dominated by product identity and specifications (`model`, `engineSize`, `mpg`), with depreciation still relevant but no longer a primary driver.

---

### 4) Interactive Analytics Interface 
To connect modeling with practical usage, we implemented an interactive interface where users can:
- enter car attributes manually (single prediction), or
- upload a CSV with multiple cars (batch prediction),

and receive predicted prices using the final deployed pipeline.

This demonstrates deployment-readiness and ensures the model can be used as an operational decision-support tool rather than only as a Kaggle artifact.

---

### 5) Ablation / Addition Study: Measuring the Value of Each Pipeline Block
To quantify the contribution of each design choice, we ran a controlled study across models using five standardised variants:  
1. **Baseline:** all original features, no feature engineering (FE) / feature selection (FS), raw price.
2. **Simplified + Log target:** drop `previousOwners`, predict `log1p(price)`.
3. **+ Depreciation proxy:** variant 2 + replace `year` with `age`.
4. **+ Feature selection:** variant 3 + FS keeping 80% of features.
5. **Full pipeline:** variant 4 but + all engineered features, FS keeping 65% of features.

Each variant was evaluated under the same cross-validation framework, enabling direct attribution of performance changes to specific pipeline steps, helping us understand which steps consistently improve generalisation (and which add complexity without measurable benefit), supporting a transparent justification of the final design.

**BEST FINAL MODEL:** Stacking Regressor (HGB + RF + ET → Ridge meta-model), variant 2 (drop `previousOwners`, predict `log1p(price)`). 

---

## Deployment
The final pipeline was retrained on the full training dataset using the selected configuration and applied to the test set to generate predictions for Kaggle submission.  
This section reproduces the same preprocessing and modeling steps used during cross-validation, ensuring consistency between offline evaluation and online performance.

---

## Conclusions
- Non-linear ensemble methods substantially outperformed linear baselines for car price prediction.
- Careful control of preprocessing and validation is as impactful (if not more) as model choice.
- HistGradientBoosting provided the best balance between accuracy and robustness, but it was when combined with other models (in a Stacking Regressor) that it truly helped us achieve our best performance.

Overall, this project shows that methodological discipline and systematic experimentation are more valuable than model complexity alone when building reliable predictive systems.
