# worldquant_data_science_lab
Takeaway from the WorldQuant University Data Science Lab 

## Getting to know data

**Tidy data** has 3 rules — one variable per column, one observation per row, one value per cell

Method chaining: 
- .dropna()
- .assign(price_usd = lambda x: ...)

Handling prices: 
- .str.replace("$","", regex=False) not to treat $ as regex pattern
- .astype(float)

Read and save:
- pd.read_csv(path)
- df.to_csv(path, index=False) Always use index=False when saving a cleaned dataset

Stat
- When mean > median (right skew) the median is the more honest "typical" value

Visualizing
- fig, ax = plt.subplots() → ax.plot(...) for explicit object-oriented style
- In boxplot, whiskers extend to the smallest value within 1.5xIQR or Q1 and to the largest value within 1.5xIQR of Q3

Correlation 
- Pearson Correlation Coefficient (r_xy): 1 strong positive linear relationship, >= 0.7 strong, >= 0.4 moderate, >=0.1 weak, below 0.1 has no meaningful linear relationship
-  series_a.corr(series_b)
-  df.corr() gets correlation matrix
-  segmented correlations

## ML Prediction

Handle multiple files
- from glob import glob
- datasets = glob("./data/name-pattern-*.csv")

Pipe Method for chaining custom function
- pandas function
- (def1(data).pipe(def2).pipe(def3)...)

Missing value visualization
- import missingno as msno
- msno.matrix(df, color=(0.3, 0.4, 0.8), ax=ax)

One-hot encoding
- Instead of assigning integers to different cities  (Palermo as 1, Recoleta as 2)
- One binary column of 0 or 1 per unique category (if dataset has 5 different cities, it creates 5 columns)
- .assign(is_palermo = lambda x: (x['neighborhood']=='Palermo').astype(int))

Train/Test split
- from sklearn.model_selection import train_test_split
- X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Baseline Performance
- from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
- mae_baseline = mean_absolute_error(y_test, [y_train.mean()]* len(y_test))
- use training mean when making prediction, not test mean

Sklearn fit/predict pattern
- Step 1: Instantiate — create the model object with hyperparameters **model = LinearRegression()**
- Step 2: Fit — train on labeled data (model learns β₀ and β₁) **model.fit(X_train, y_train)**
- Step 3: Predict — generate predictions for any feature matrix **y_pred = model.predict(X_test)** 
- Preserve y_test only for evaluation

Evaluation metrics
- MAE: average magnitude of prediction error
- RMSE: large errors are penalized disproportionately
- R^2 Coefficient of determination: how much better the model is than always predicting the mean. 1.0 means perfect prediction, 0 means similar to baseline mean, < 0 worse.

Diagnose the model
- Overfitting (train ≪ test error)
- Good Fit (train ≈ test, both reasonably low)

Regularization to prevent overfitting
- Ridge L2 and Lasso L1
- Feature scaling gets mean = 0 and std = 1
- Pipeline to prevent data leakage


model = make_pipeline(

    OneHotEncoder(use_cat_names=True),   # Step 1: encode categories
    
    StandardScaler(),                     # Step 2: scale to mean=0, std=1
    
    Ridge(alpha=1.0)                      # Step 3: fit regularized model // for L1 Lasso(alpha=100, max_iter=10000)
    
)

model.fit(X_train, y_train)   # All transformers fit ONLY on X_train

model.predict(X_test)  

- Ridge: shrink coefficient close to zero
- Lasso: coefficient can reach 0 and performs automatic feature selection. The model that emerges is sparse, most coefficients are zero, and only the truly important features retain non-zero coefficients

Alpha
- controls a fundamental tradeoff between bias (model misses real patterns) and variance (model has too much noise)
- alpha approaches 0: No regularization — model is ordinary linear regression. Risk of high variance (overfitting)
- alpha approaches infinity: Maximum regularization — all coefficients driven to zero. Risk of high bias (underfitting)
- The diagnostic rule: If train RMSE ≪ test RMSE → increase alpha (reduce variance) to add more regularization

Automated Alpha Selection
- cross-validation
- RidgeCV and LassoCV

Model Evaluation
- Build DRY metric function
- def get_rmse(model, X, y_true):
    y_pred = model.predict(X)
    return np.sqrt(mean_squared_error(y_true, y_pred))
- Small gap between RMSE train and RMSE test means good generalization
- Plots: predict vs actual, residual
- Heteroscedasticity (non-constant error variance): small residual at low prices, large spread at high prices
- best_model = df_results["Test_RMSE"].idxmin()
  
