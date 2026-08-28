import pandas as pd
import numpy as np

# DECISION TREE IMPORTS

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# LOAD CSV FILE

df = pd.read_csv("battery_SOC_SOH_output.csv")

# BATTERY PARAMETERS

RATED_CAPACITY_AH = 15
MANUAL_CYCLE_COUNT = 20

V_MAX = 4.2
V_MIN = 2.5

NORMAL_TEMP = 25

# CLEAN DATA

df.dropna(inplace=True)

df.reset_index(
    drop=True,
    inplace=True
)

# INITIAL SOC USING NONLINEAR OCV

initial_voltage = df.loc[
    0,
    'filtered_voltage'
]

initial_soc = (
    (
        (initial_voltage - V_MIN)
        / (V_MAX - V_MIN)
    ) ** 0.5
) * 100

initial_soc = np.clip(
    initial_soc,
    0,
    100
)

print(
    f"Initial SOC = {initial_soc:.2f}%"
)

# KALMAN FILTER SOC ESTIMATION

df['Real_SOC'] = 0.0

df.loc[0, 'Real_SOC'] = initial_soc

# KALMAN FILTER PARAMETERS

Q = 0.001
R = 50
P = 1.0

# KF LOOP

for i in range(1, len(df)):

    current = abs(
        df.loc[i, 'filtered_current']
    )

    voltage = df.loc[
        i,
        'filtered_voltage'
    ]

    dt_seconds = df.loc[
        i,
        'dt'
    ]

    dt_hours = dt_seconds / 3600

    # PREDICTION STEP

    soc_drop = (
        current * dt_hours
        / RATED_CAPACITY_AH
    ) * 100

    soc_pred = (
        df.loc[i - 1, 'Real_SOC']
        - soc_drop
    )

    soc_pred = np.clip(
        soc_pred,
        0,
        100
    )

    P = P + Q

    # MEASUREMENT STEP

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
        (voltage_ratio ** 0.5)
        * 100
    )

    # KALMAN GAIN

    K = P / (P + R)

    # UPDATE STEP

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

    df.loc[
        i,
        'Real_SOC'
    ] = soc_updated

# FINAL SOC

final_soc = df[
    'Real_SOC'
].iloc[-1]

print(
    f"Final Real SOC = {final_soc:.2f}%"
)


# SOH ESTIMATION  

cycle_degradation = (
    MANUAL_CYCLE_COUNT
    * 0.00002
)

avg_temp = df[
    'temperature'
].mean()

temperature_degradation = max(
    0,
    (avg_temp - NORMAL_TEMP)
) * 0.001

# THERMAL EXPANSION

THERMAL_EXPANSION_COEFF = 0.0000001

df['Thermal_Expansion'] = 0.0

total_thermal_expansion = 0

for i in range(len(df)):

    current_temp = df.loc[
        i,
        'temperature'
    ]

    dt_seconds = df.loc[
        i,
        'dt'
    ]

    temp_difference = abs(
        current_temp - NORMAL_TEMP
    )

    thermal_expansion = (
        temp_difference
        * THERMAL_EXPANSION_COEFF
        * dt_seconds
    )

    df.loc[
        i,
        'Thermal_Expansion'
    ] = thermal_expansion

    total_thermal_expansion += thermal_expansion

print(
    f"Total Thermal Expansion = "
    f"{total_thermal_expansion:.6f}"
)

# TOTAL DEGRADATION

total_degradation = (
    cycle_degradation
    + temperature_degradation
    + total_thermal_expansion
)

total_degradation = min(
    total_degradation,
    0.4
)

# ESTIMATED CAPACITY

estimated_capacity = (
    RATED_CAPACITY_AH
    * (1 - total_degradation)
)

print(
    f"Estimated Capacity = "
    f"{estimated_capacity:.3f} Ah"
)

df['Total_Thermal_Expansion'] = (
    total_thermal_expansion
)

# SOH CALCULATION

soh = (
    estimated_capacity
    / RATED_CAPACITY_AH
) * 100

soh = np.clip(
    soh,
    0,
    100
)

print(
    f"SOH = {soh:.2f}%"
)

df['SOH'] = soh

# CAPACITY FADE

capacity_fade_percent = (
    100 - soh
)

df['Capacity_Fade_Percent'] = (
    capacity_fade_percent
)

#  
# DECISION TREE REGRESSOR
#  

print(
    "\nTraining Decision Tree Model..."
)

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

# CREATE MODEL

dt_model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

# TRAIN MODEL

dt_model.fit(
    X_train,
    y_train
)

# PREDICT TEST DATA

y_pred = dt_model.predict(
    X_test
)

# PREDICT FULL DATASET

df['ML_SOC'] = dt_model.predict(X)

df['ML_SOC'] = np.clip(
    df['ML_SOC'],
    0,
    100
)

print(
    "ML SOC Prediction Completed"
)

# SAVE OUTPUT

df.to_csv(
    "battery_final_output.csv",
    index=False
)

print(
    "\nSOC and SOH calculated successfully "
    "using Kalman Filter + Decision Tree."
)

print(
    "\nFinal output saved as battery_final_output.csv"
)