import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
#--------------Step 1: Load & Manipulate Data ----------------
#Load dataset
df = pd.read_csv("california_housing.csv")

print("Showing first 5 rows:\n",df.head())
print("Dataset shape(rows, columns):",df.shape)

#Check missing value
print("Missing values:\n",df.isnull().sum())
df.drop_duplicates(inplace=True)

#--------------Step 2: Exploratory Data Analysis --------------
#print("Descriptive statistics:\n",df.describe(),"median",df.median())
describe=df.describe()
describe.loc['median'] = df.median()
print("Descriptive statistics:\n",describe)

#Visualiztions
#Histogram of Price
plt.figure(figsize=(6,5))
sns.histplot(df['Price'], bins=30, kde=True)
plt.title("Histogram of Price")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

#Scatter Plot for income vs price
plt.figure(figsize=(6,5))
sns.scatterplot(x=df['Income'],y=df['Price'])
plt.title("Scatter Plot: income vs price")
plt.xlabel("Income")
plt.ylabel("Price")
plt.show()

#Correlation heatmap of all features
corr=df.corr()
print("Correlation of all features:\n",corr)
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True,cmap="viridis")
plt.title("Correlation heatmap of all features")
plt.xticks(rotation=90)
plt.show()


#Interpret findings in words
print("\nEDA interpretation hints:")
print("- Check skew of Price from histogram.")
print("- Look for positive correlation Income <-> Price in scatter.")
print("- Use correlation matrix to see likely strong predictors (abs(corr) high).")

#------------Step 3: Preprocessing & Outlier Detection --------------
#Scale numeric features

features = ["Income","Age","Rooms","Bedrooms","Population","Occupancy","Latitude","Longitude"]
scaler= StandardScaler()
scaled=df.copy()
scaled[features]=scaler.fit_transform(scaled[features])
print("Scale numeric features:\n",scaled[features])

#Outlier detection
df_no = df.copy()
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print("IQR:", IQR)
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
outlier = ((df < (lower)) | (df > (upper))).sum()
print("Outliers in each column:\n", outlier)

#boxplot for incoome and rooms
plt.figure(figsize=(6,5))
sns.boxplot(df[['Income','Rooms']])
plt.title("Boxplot of income & rooms")
plt.show()


#drop extreme outlier
df_no = df[~((df < (lower)) | (df > (upper))).any(axis=1)]
print("columns without outliers:",df.shape)
print("Data shape after removing outliers:", df_no.shape)

#------------- Step 4: Train-Test Split -----------------
#Random split(80-20)
X_no_outlier = df_no.drop('Price', axis=1)
y_no_outlier= df_no['Price']
X_train, X_test, y_train, y_test = train_test_split(X_no_outlier, y_no_outlier, test_size=0.2, random_state=42)
print("Random Training set shape:", X_train.shape)
print("Random Test set shape:", X_test.shape)

#Stratified split based on income groups (bins)
# Create income categories for stratification
X_no_outlier['Income_bin'] = pd.cut(X_no_outlier['Income'],
                         bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                         labels=[1, 2, 3, 4, 5])
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in strat_split.split(X_no_outlier, X_no_outlier['Income_bin']):
    X_train_s = X_no_outlier.iloc[train_index]
    y_train_s = y_no_outlier.iloc[train_index]
    X_test_s = X_no_outlier.iloc[test_index]
    y_test_s = y_no_outlier.iloc[test_index]

# Drop the income category from the split data


X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_no_outlier, y_no_outlier, test_size=0.2, stratify=X_no_outlier["Income_bin"], random_state=42)

# Safe drop
X_train_s = X_train_s.drop("Income_bin", axis=1).copy()
X_test_s = X_test_s.drop("Income_bin", axis=1).copy()

print("\nStratified Split Train shape:", X_train_s.shape)
print("Stratified Split Test shape:", X_test_s.shape)


#Compare distributions
print("\nRandom Train Income Distribution:\n", pd.cut(X_train["Income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5]).value_counts(normalize=True))
print("Stratified Train Income Distribution:\n", pd.cut(X_train_s["Income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5]).value_counts(normalize=True))
print("\nRandom Test Income Distribution:\n", pd.cut(X_test["Income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5]).value_counts(normalize=True))
print("Stratified Test Income Distribution:\n", pd.cut(X_test_s["Income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5]).value_counts(normalize=True))


#-----------Step 5: Regression Models -------------------
# Simple Regression: Income -> Price
model_simple = LinearRegression()
model_simple.fit(X_train[['Income']], y_train)
y_pred_simple = model_simple.predict(X_test[['Income']])

# Multiple Regression: all features
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)
y_pred_multi = model_multi.predict(X_test)

# Equations (coeffs)
print("\nSimple regression: Intercept =", model_simple.intercept_, "Coeff(Income) =", model_simple.coef_[0])
print("\nMultiple regression coefficients:")
for f, c in zip(features, model_multi.coef_):
    print(f, ":", c)
print("Intercept:", model_multi.intercept_)

# ---------------- Step 6: Assumptions Checks ----------------
# Residuals
resid_multi = y_test - y_pred_multi

# Linearity: Predicted vs Actual
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred_multi, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='black')
plt.title("Predicted vs Actual (Multiple Regression)")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# Independence: residuals vs order
plt.figure(figsize=(8,3))
plt.plot(resid_multi.values)
plt.title("Residuals vs Order")
plt.xlabel("Test index order")
plt.ylabel("Residual")
plt.show()

# Homoscedasticity: predicted vs residuals
plt.figure(figsize=(6,4))
plt.scatter(y_pred_multi, resid_multi, s=10)
plt.axhline(0)
plt.title("Predicted vs Residuals")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.show()

# Normality: histogram and Q-Q
plt.figure(figsize=(6,3))
plt.hist(resid_multi, bins=30)
plt.title("Histogram of residuals")
plt.show()

plt.figure(figsize=(6,4))
stats.probplot(resid_multi, dist="norm", plot=plt)
plt.title("Q-Q plot of residuals")
plt.show()

# ---------------- Step 7: Metrics ----------------
mae_simple= mean_absolute_error(y_test, y_pred_simple)
mse_simple= mean_squared_error(y_test, y_pred_simple)
rmse_simple = np.sqrt(mse_simple)
r2_simple = r2_score(y_test, y_pred_simple)

mae_multi = mean_absolute_error(y_test, y_pred_multi)
mse_multi = mean_squared_error(y_test, y_pred_multi)
rmse_multi = np.sqrt(mse_multi)
r2_multi = r2_score(y_test, y_pred_multi)
print("Metrics for simple regression:\n")
print("MAE:", mae_simple)
print("MSE:", mse_simple)
print("RMSE:", rmse_simple)
print("R²:", r2_simple)

print("\nMetrics for multi regression:\n")
print("MAE:", mae_multi)
print("MSE:", mse_multi)
print("RMSE:", rmse_multi)
print("R²:", r2_multi)
# Simple interpretation
print(f"\nOn average, our predictions are off by {mae_multi:.2f}.")
print(f"The model explains {r2_multi*100:.2f}% of the variation in house prices.")

# ---------------- Step 8: Final Report Hints ----------------

print("\n 1. Which variables impact house price the most?\n Usually Income has the strongest effect, followed by Rooms and Location.")

print("\n2. Did assumptions hold true?\n Mostly yes, but small deviations may exist (e.g., slight non-normality in residuals")

print("\n3. Which model was better?\n Multiple Regression performed better than Simple Regression (higher R², lower errors")

print("\n4. Are errors acceptable?\nYes, because RMSE/MAE are reasonably small compared to the price range")
