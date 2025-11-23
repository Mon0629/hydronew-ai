import numpy as np
import pandas as pd

np.random.seed(42)
n = 10000
safe_ratio_target = 0.5  # roughly 50% Safe

# Water treatment thresholds
pH_safe_min, pH_safe_max = 6.5, 8.0
tds_safe_upper = 1500
turbidity_safe_upper = 5

# Noise to avoid deterministic boundaries
ph_noise = 0.2
tds_noise = 50
turbidity_noise = 0.5

# Generate features
ph = np.random.uniform(4.0, 9.5, n) + np.random.normal(0, ph_noise, n)
tds = np.random.uniform(100, 5000, n) + np.random.normal(0, tds_noise, n)
turbidity = np.random.uniform(0.1, 50, n) + np.random.normal(0, turbidity_noise, n)

# Clip to plausible ranges
ph = np.clip(ph, 0, 14)
tds = np.clip(tds, 0, 5000)
turbidity = np.clip(turbidity, 0, 1000)

# Label generation
label = (
    (ph >= pH_safe_min) & (ph <= pH_safe_max) &
    (tds <= tds_safe_upper) &
    (turbidity <= turbidity_safe_upper)
).astype(int)

# Adjust Safe/Unsafe ratio
safe_count = label.sum()
desired_safe_count = int(n * safe_ratio_target)

if safe_count < desired_safe_count:
    unsafe_indices = np.where(label == 0)[0]
    flip_indices = np.random.choice(unsafe_indices, desired_safe_count - safe_count, replace=False)
    label[flip_indices] = 1
elif safe_count > desired_safe_count:
    safe_indices = np.where(label == 1)[0]
    flip_indices = np.random.choice(safe_indices, safe_count - desired_safe_count, replace=False)
    label[flip_indices] = 0

# Assemble dataframe
df = pd.DataFrame({
    'ph': ph,
    'turbidity': turbidity,
    'tds': tds,
    'status': label
})

df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('data/external/synthetic_data.csv', index=False)

print(df.head())
print(f"Safe samples: {df['status'].sum()}, Unsafe samples: {n - df['status'].sum()}")
