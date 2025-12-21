# Cars4You 

## Abstract 
- Cars4You is a car resale company that reached out in hopes to obtain a faster way to get price estimations for their cars. Our main goals were to find a model that helps with those predictions and, additionally, an interface that employees can use to send either CSV with many cars or to predict the price for a car by just filling a form. 

- We started by analysing the dataset provided. We conducted a __univariate analysis__, by seeing individual statistics of the features, such as mean, median, maximum and minimum values (for numerical variables) and counts, frequencies, accessing categories (for ctegorical), as well as analysing box plots, histograms, among other visualizations. We did some __multivariate analysis__ to better understand relationships between features and correlation. 

- Based on what we saw in the __EDA__, we developed our __preprocessing functions__, that were based on a __fit and transform logic__ whenever we need to immpute statistcis or values that would include data from validation. And for deterministic transfromations (such as replacing missing values with unknown, removing spaces, put all letters in uppercase, limit strings to alpha numeric characters, etc) was done to the full dataset, in order to be more efficient instead of apllying all of these transformations per fold. We then apllied all of these functions to the full dataset in eda notebook and compared to the original, so we could visualise the chnages we were explainig.

- We created __some enginnerred features__ (age, owners_flagged - 1 if previousOwners > 1, else 0 -, log_miles_per_year, creating engine bins). Performed scaling (using StandardScaler) and divided the encoding strategy into 2: target encoding for high cardinality features and one hot encoding for the remaining categorical features. 

- The models developed were, from the desion trees family, Random Forest Regressor, Histogram Gradient Boosting, Extra Trees,  Bagging Regressor (Decision Trees); from linear family, OLS and Regularised Linear; neural networks MLP, Stacking Regressor (HGB + RF + ET -> Ridge meta-model) and, although not a model per say Weighted Average (HGB + RF). 

- For each model, we accessed its performance within an __8-fold cross validation__ structure and performed random search in order to find the best configurations for the model. For each set of 8 folds we would test it under a random configuration sampled from our dictionary with all hyperparameter possible values. We would then print the average of the error metrics for the folds (RMSE and MAE were always present since they were our main error metrics, and bias an r2 were often printed just for adittional check and in case it was later needed). RMSE was our primary error metric indicator, since it penalizes more heavlily big erros and, in this context where the selling prices are high, making a big mistake is far worse for business. In case of tie or very RMSE similar values, we saw the MAE score as well. 

- Across all models we tested them under different settings such as using all features (without paintQuality% and hasDamage), all features but excluding previousOwners and price log transformed, all features and engineered features with a Feature Selection Random Forest set to 65% of most important, all original features without previousOwners and with age and the same as this one but with feature selection set to 80%. 

- We then compared all models and concluded that .... Stacking got the best result...

- In conclusion, we are very pleasent with the result we got, we have a rubost model predictior using Stacking (combining HGB + RF + ET -> Ridge meta-model) with reasonable RMSE that will aid Cars4You expedicting their cars faster. We were able, as well, to develop the interface for an easier acess for the company employees. 
- We started by analysing the dataset provided. We conducted a univariate analysis, by seeing individual statistics of the features, such as mean, median, maximum and minimum values (for numerical variables), counts and frequencies (for categorical), as well as analysing box plots, histograms, among other visualizations. We did some multivariate analysis to better understand relationships between features and correlation.


## Project Structure 


## Group Member Contributions 
- Francisca Fernandes (20250406): 33%
__Specific Contributions:__ Univariate Analysis, Numerical Variables Preprocessing, 8-Fold Cross Validation Structure, Feature Engineering, Development of RandomForest and HistGradientBoosting Regressors, Prediction Interface 
- Maria Pimentel (20250466): 33%
__Specific Contributions:__ Multivariate Analysis, Scaling, ExtraTrees and Linear Models, Distribution Shift Between Training and Test Data, Model Visualizations, Feature Importances per Price Segment
- Mariana Melo (20250414): 33% 
__Specific Contributions:__ Categorical Data Preprocessing, Feature Selection, Encoding, MLP and Ensemble Models (Bagging, Weighted Average and Stacking)
Specific Contributions: Categorical Data Preprocessing, Feature Selection, Encoding, MLP and Ensemble Models (Bagging, Weighted Average and Stacking)
> Even though these were the main contributions of each member, it is importante to note that every decision was made based on discussions between everyone.
