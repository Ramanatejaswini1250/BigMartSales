import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor

# Load the data
df = pd.read_csv('data/train.csv')

# Fill missing values
df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
df['Item_Visibility'] = df['Item_Visibility'].fillna(df['Item_Visibility'].median())
df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])

# Drop non-informative identifier columns
test_features = df.drop(columns=['Item_Identifier', 'Outlet_Identifier'])

# Encode categorical columns
categorical_columns = ['Item_Fat_Content', 'Outlet_Location_Type', 'Item_Type', 'Outlet_Type', 'Outlet_Size']
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_data = encoder.fit_transform(df[categorical_columns])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))

# Merge encoded columns
df_encoded = df.drop(columns=categorical_columns + ['Item_Identifier', 'Outlet_Identifier']).reset_index(drop=True)
df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

# Split into features and target
X = df_encoded.drop(columns=['Item_Outlet_Sales'])
y = df_encoded['Item_Outlet_Sales']

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric columns
numeric_columns = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']

# Recheck that these columns exist in df_encoded before scaling
missing_columns = [col for col in numeric_columns if col not in df_encoded.columns]
if missing_columns:
    print(f"Warning: Missing columns in df_encoded: {missing_columns}")

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_val_scaled[numeric_columns] = scaler.transform(X_val[numeric_columns])

# Define models
def evaluate_model(model, name):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"{name} RMSE: {rmse:.4f}")
    return name, rmse, y_pred

# Evaluate all models
results = []
results.append(evaluate_model(LinearRegression(), 'Linear Regression'))
results.append(evaluate_model(Ridge(alpha=1.0), 'Ridge Regression'))
results.append(evaluate_model(Lasso(alpha=0.1, max_iter=10000), 'Lasso Regression'))
results.append(evaluate_model(RandomForestRegressor(n_estimators=100, random_state=42), 'Random Forest'))
results.append(evaluate_model(xgb.XGBRegressor(objective='reg:squarederror', random_state=42), 'XGBoost'))
results.append(evaluate_model(LGBMRegressor(random_state=42), 'LightGBM'))

# Save results to CSV
results_df = pd.DataFrame(results, columns=["Model", "RMSE", "Predictions"])
results_df.drop(columns=["Predictions"], inplace=True)  # Drop predictions from saved CSV
results_df.to_csv("data/model_comparision_submission.csv", index=False)
print("Model evaluation results saved to model_comparision_submission.csv")

# Plot RMSEs
model_names, rmse_values, _ = zip(*results)
plt.figure(figsize=(10, 6))
sns.barplot(x=list(model_names), y=list(rmse_values))
plt.title('Model Comparison (RMSE)')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot Actual vs Predicted for best model
best_model_name, _, best_pred = sorted(results, key=lambda x: x[1])[0]
print(f"Best Model: {best_model_name}")

# Create new data for prediction (in this case, using df, excluding Item_Outlet_Sales)
new_data = df[['Item_Weight', 'Item_Visibility', 'Outlet_Size', 'Item_Fat_Content',
               'Outlet_Location_Type', 'Item_Type', 'Outlet_Type',
               'Item_Identifier', 'Outlet_Identifier','Item_MRP', 'Outlet_Establishment_Year']].copy()

new_encoded_data = encoder.transform(df[categorical_columns])
new_encoded_df = pd.DataFrame(new_encoded_data, columns=encoder.get_feature_names_out(categorical_columns))

# Ensure that we only include columns that are present in the DataFrame
required_columns = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']
existing_columns = [col for col in required_columns if col in new_data.columns]

# Create the new encoded DataFrame by concatenating only the existing columns
new_df_encoded = pd.concat([new_data[existing_columns], new_encoded_df], axis=1)

# Align columns in new_df_encoded with X_train_scaled
new_df_encoded_aligned = new_df_encoded[X_train_scaled.columns]

# Scale numeric columns using the previously fitted scaler
new_df_encoded_scaled = new_df_encoded_aligned.copy()

# Ensure only existing numeric columns are scaled
existing_numeric_columns = [col for col in numeric_columns if col in new_df_encoded.columns]
if existing_numeric_columns:
    new_df_encoded_scaled[existing_numeric_columns] = scaler.transform(new_df_encoded_aligned[existing_numeric_columns])
else:
    print("No numeric columns found for scaling in new_df_encoded.")

# Define model lookup
model_lookup = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1, max_iter=10000),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
    'LightGBM': LGBMRegressor(random_state=42)
}

# Train best model
best_model = model_lookup[best_model_name]
best_model.fit(X_train_scaled, y_train)

# Proceed with prediction (use new_df_encoded_scaled if scaling was successful)
if existing_numeric_columns:
    X_new = new_df_encoded_scaled  # No need to drop the identifier columns now
    new_predictions = best_model.predict(X_new)
    # Create output DataFrame with selected columns
    output_df = new_data[['Item_Identifier', 'Outlet_Identifier']].copy()  # Keep original identifiers
    output_df['Item_Outlet_Sales'] = new_predictions
    # Save predictions to CSV
    output_df.to_csv('data/predicted_sales.csv', index=False)
    print("Predictions with selected columns saved to predicted_sales.csv")
else:
    print("Skipping prediction due to missing numeric columns.")

# Plot Actual vs Predicted Sales
plt.figure(figsize=(10, 6))
sns.regplot(x=y_val, y=best_pred, scatter_kws={'s': 30}, line_kws={'color': 'red'})
plt.title(f'Actual vs Predicted Sales ({best_model_name})')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.tight_layout()
plt.show()

# Residuals
residuals = y_val - best_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30, color='skyblue')
plt.title(f'Residuals Distribution ({best_model_name})')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
