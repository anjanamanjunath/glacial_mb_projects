import pandas as pd
import numpy as np
import cartopy
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

import tensorflow as tf
import os

from sklearn import ensemble


all_data = pd.read_csv('/**/data/all_data.csv', index_col=['rgi_id', 'period'])

# setting up test/train split
g_idx = np.unique(all_data.index.get_level_values(0).values)

g_train, g_test = train_test_split(g_idx, train_size=0.2,test_size=0.8)

train_df = all_data.loc[g_train]
test_df = all_data.loc[g_test]

testing_df = test_df.to_csv('/**/data/df_test.csv')

g_idx = np.unique(train_df.index.get_level_values(0).values)

train_dataset, validation_dataset = train_test_split(g_idx, train_size=0.85, test_size=0.15)

train_df = all_data.loc[train_dataset]
val_df = all_data.loc[validation_dataset]

features_to_drop = ['dmdtda', 'err_dmdtda', 'target_id']

train_X = train_df.drop(features_to_drop, axis=1)
train_Y = train_df[['dmdtda']]

train_X, train_Y = train_X.values, train_Y.values
train_Y = np.nan_to_num(train_Y)

val_X = val_df.drop(features_to_drop, axis=1)
val_Y = val_df[['dmdtda']]

scaler = StandardScaler()
scaler.fit(train_X)

X_train_scaled = scaler.transform(train_X)
X_validation_scaled = scaler.transform(val_X)

test_X = test_df.drop(features_to_drop, axis=1)
test_Y = test_df[['dmdtda']]

test_X, test_Y = test_X.values, test_Y.values
test_Y = np.nan_to_num(test_Y)

# MLR 
elastic_model = ElasticNet(alpha=0, l1_ratio=0, fit_intercept=True) 
elastic_model.fit(X_train_scaled, train_Y)
elastic_model.predict(X_train_scaled)

# Random Forest 
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
bootstrap = [True, False]

param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'bootstrap': bootstrap}

rf = ensemble.RandomForestRegressor()
grid_sr = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, verbose=2, n_jobs = -1)

grid_sr.fit(train_X, train_Y)
grid_sr.best_params_

params = {'bootstrap': True,
 'max_depth': None,
 'max_features': 'auto',
 'min_samples_split': 5,
 'n_estimators': 200}

reg_ensemble = ensemble.RandomForestRegressor(**params)

reg_ensemble.fit(train_X, train_Y)
reg_ensemble.predict(X_validation_scaled)

# feature importance plot 

feature_importance = reg_ensemble.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, np.array(val_X.columns)[sorted_idx])
plt.title("Feature Importance (MDI)")

plt.show()


# Make run determistic
tf.keras.backend.clear_session()
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Configure model architecture
input = tf.keras.layers.Input((34,))

x = tf.keras.layers.Dense(200, activation=tf.nn.relu)(input)
x = tf.keras.layers.Dropout(0.3)(x)

x = tf.keras.layers.Dense(150, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dropout(0.1)(x)

x = tf.keras.layers.Dense(100, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dropout(0.1)(x)

x = tf.keras.layers.Dense(50, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dropout(0.1)(x)

x = tf.keras.layers.Dense(50, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dropout(0.1)(x)

x = tf.keras.layers.Dense(50, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dropout(0.1)(x)

x = tf.keras.layers.Dense(50, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dropout(0.1)(x)

output = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=input, outputs=output)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(loss='mae', optimizer=optimizer, metrics=['mse'])

# Run model
ann_run = model.fit(train_X, train_Y, batch_size=64,epochs=1000, verbose=2, validation_split=0.1)

ypred = model.predict(test_X)
ypred = np.squeeze(ypred)

print(model.evaluate(train_X, train_Y))

mae = np.mean(abs(ypred - test_Y))
rmse = mean_squared_error(test_Y, ypred, squared=False)
r2 = metrics.r2_score(test_Y, ypred) 

print('MAE:',mae)
print('RMSE', rmse)
print('R2:',r2)

model.summary()











