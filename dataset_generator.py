import numpy as np
import pandas as pd


np.random.seed(42)

n = 10000


pH = np.random.uniform(4.0, 9.5, n)
turbidity = np.random.uniform(0.1, 50, n)  
tds = np.random.uniform(100, 5000, n)
conductivity = np.random.uniform(200, 10000, n)

label = (
    (pH >= 6.5) & (pH <= 8.0) &
    (turbidity < 5) &
    (tds < 1500) &
    (conductivity < 3000)
).astype(int)


df = pd.DataFrame({
    'pH': pH,
    'turbidity': turbidity,
    'TDS': tds,
    'conductivity': conductivity,
    'label': label
})


df.to_csv('data/external/synthetic_data.csv', index=False)

print(df.head())
