import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

  
# LOAD DATA
  

df = pd.read_csv("battery_SOC_SOH_output.csv")

  
# BATTERY PARAMETERS
  

RATED_CAPACITY_AH = 15
CYCLE_COUNT = 20

V_MAX = 4.2
V_MIN = 2.5
NORMAL_TEMP = 25

  
# CLEAN DATA
  

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

  
# INITIAL SOC USING OCV
  

initial_voltage = df.loc[0, 'filtered_voltage']

initial_soc = (
    np.sqrt(
        (initial_voltage - V_MIN)
        / (V_MAX - V_MIN)
    )
) * 100

initial_soc = np.clip(initial_soc, 0, 100)

  
# KALMAN FILTER SOC
  

df['Real_SOC'] = 0.0
df.loc[0, 'Real_SOC'] = initial_soc

Q = 0.001
R = 50
P = 1.0

for i in range(1, len(df)):

    current = abs(df.loc[i, 'filtered_current'])
    voltage = df.loc[i, 'filtered_voltage']

    dt_hours = df.loc[i, 'dt'] / 3600

    # Coulomb Counting

    soc_drop = (
        current * dt_hours
        / RATED_CAPACITY_AH
    ) * 100

    soc_pred = (
        df.loc[i-1, 'Real_SOC']
        - soc_drop
    )

    soc_pred = np.clip(
        soc_pred,
        0,
        100
    )

    P = P + Q

    # Voltage-based SOC

    voltage_ratio = (
        (voltage - V_MIN)
        / (V_MAX - V_MIN)
    )

    voltage_ratio = np.clip(
        voltage_ratio,
        0,
        1
    )

    soc_measured = (
        np.sqrt(voltage_ratio)
        * 100
    )

    # Kalman Gain

    K = P / (P + R)

    # Update

    soc_updated = (
        soc_pred
        + K * (
            soc_measured - soc_pred
        )
    )

    soc_updated = np.clip(
        soc_updated,
        0,
        100
    )

    P = (1 - K) * P

    df.loc[i, 'Real_SOC'] = soc_updated

  
# FINAL SOC
  

final_soc = df['Real_SOC'].iloc[-1]

print(
    f"Final SOC = {final_soc:.2f}%"
)

  
# SOH CALCULATION
  

cycle_degradation = (
    CYCLE_COUNT * 0.00002
)

avg_temp = df['temperature'].mean()

temperature_degradation = (
    max(
        0,
        avg_temp - NORMAL_TEMP
    ) * 0.001
)

total_degradation = (
    cycle_degradation
    + temperature_degradation
)

total_degradation = min(
    total_degradation,
    0.4
)

estimated_capacity = (
    RATED_CAPACITY_AH
    * (1 - total_degradation)
)

soh = (
    estimated_capacity
    / RATED_CAPACITY_AH
) * 100

soh = np.clip(
    soh,
    0,
    100
)

capacity_fade = 100 - soh

df['SOH'] = soh
df['Capacity_Fade'] = capacity_fade

print(
    f"SOH = {soh:.2f}%"
)

  
# SVR MODEL
  

X = df[[
    'filtered_voltage',
    'filtered_current',
    'temperature',
    'dt'
]]

y = df['Real_SOC']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

X_scaled = scaler.transform(
    X
)

# SVR

svr_model = SVR(
    kernel='rbf',
    C=100,
    epsilon=0.1
)

svr_model.fit(
    X_train_scaled,
    y_train
)

# Predictions

y_pred = svr_model.predict(
    X_test_scaled
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)

print(
    f"MAE = {mae:.4f}"
)

print(
    f"R2 Score = {r2:.4f}"
)

# Full Dataset Prediction

df['ML_SOC'] = svr_model.predict(
    X_scaled
)

df['ML_SOC'] = np.clip(
    df['ML_SOC'],
    0,
    100
)

  
# SAVE OUTPUT
  

df.to_csv(
    "battery_SVR_output.csv",
    index=False
)

print(
    "\nSOC, SOH and SVR prediction completed."
)

print(
    "Output saved as battery_SVR_output.csv"
)