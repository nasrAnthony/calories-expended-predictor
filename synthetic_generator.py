import numpy as np
import pandas as pd

existing_df = pd.read_csv('.\\Training-data\\data-book.csv')

existing_df['Intensity'] = np.nan
max_user_id = existing_df['User_ID'].max()

n_samples = 4000

np.random.seed(42)
age = np.random.randint(18, 60, n_samples)
weight = np.random.normal(70, 15, n_samples).clip(40, 150).astype(int)
height = np.random.normal(170, 10, n_samples).clip(140, 200).astype(int)
gender = np.random.choice(['male', 'female'], n_samples)
heart_rate = np.random.uniform(60, 180, n_samples).astype(int) # 60 <= hr <= 180
intensity = np.random.choice([0, 1], n_samples) #0 = moderate intensity, 1 = High intensity
duration_seconds = np.random.uniform(10, 30, n_samples)
duration_minutes = duration_seconds / 60  

base_ratio = np.where(intensity == 1, 0.0880, 0.0680)

calories_burned = (base_ratio * (weight / 70) * (duration_seconds) * (heart_rate / 100))

user_ids = np.arange(max_user_id + 1, max_user_id + 1 + n_samples)

synthetic_df = pd.DataFrame({
    'User_ID': user_ids,
    'Gender': gender,
    'Age': age,
    'Height': height,
    'Weight': weight,
    'Duration': duration_minutes,
    'Heart_Rate': heart_rate,
    'Calories': calories_burned, 
    'Intensity': intensity
})

combined_df = pd.concat([existing_df, synthetic_df], ignore_index=True)

combined_df.to_csv('.\\Training-data\\combined_with_intensity_dataset.csv', index=False)

combined_df.head()