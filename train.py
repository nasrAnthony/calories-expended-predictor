import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae, r2_score
import joblib

df = pd.read_csv('.\\Training-data\\combined_with_intensity_dataset.csv')

df.describe()
sb.scatterplot(x=df['Height'], y=df['Weight'])
plt.show()

features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate']
plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    x = df.sample(1000)
    sb.scatterplot(x=x[col], y=x['Calories'])
plt.tight_layout()
plt.show()

df.replace({'male': 0, 'female': 1}, inplace=True)

imputer = SimpleImputer(strategy='constant', fill_value=0)  # Replace NaN values with 0 (low intensity)
df['Intensity'] = imputer.fit_transform(df[['Intensity']])

plt.figure(figsize=(8, 8))
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()

ftrs = df.drop(['User_ID', 'Calories'], axis=1)
target = df['Calories'].values

X_train, X_val, Y_train, Y_val = train_test_split(ftrs, target, test_size=0.2, random_state=22)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

models = [LinearRegression(), XGBRegressor(), Lasso(), RandomForestRegressor(), Ridge()]

model_names = []
training_mae_list = []
validation_mae_list = []
training_r2_list = []
validation_r2_list = []

for model in models:
    model.fit(X_train, Y_train)
    model_names.append(type(model).__name__)

    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    training_mae_list.append(mae(Y_train, train_preds))
    validation_mae_list.append(mae(Y_val, val_preds))
    training_r2_list.append(r2_score(Y_train, train_preds))
    validation_r2_list.append(r2_score(Y_val, val_preds))

results_df = pd.DataFrame({
    'Model': model_names,
    'Training MAE': training_mae_list,
    'Validation MAE': validation_mae_list,
    'Training R²': training_r2_list,
    'Validation R²': validation_r2_list
})

sorted_results_df = results_df.sort_values(by='Validation R²', ascending=False)


print(sorted_results_df)

best_model = models[3]  
joblib.dump(best_model, 'calorie_predictor_model.pkl')


loaded_model = joblib.load('calorie_predictor_model.pkl')


input_data = np.array([[0, 21, 176, 98, 0.20, 103, 1]])  #Gender, Age, Height, Weight, Duration, Heart Rate, Intensity
scaled_input = scaler.transform(input_data)
predicted_calories = loaded_model.predict(scaled_input)
print(predicted_calories)
