(iii)train_test_split.py import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('data_file.csv')

# Ensure 'benign' column exists if 'Benign' not in df.columns:
raise ValueError("'Benign' column is missing in the dataset")


# Separate benign (1) and malicious (0) df_benign = df[df['Benign'] == 1] df_malicious = df[df['Benign'] == 0]

# Find minimum count to balance
min_count = min(len(df_benign), len(df_malicious))


# Undersample both to min_count
df_benign_balanced = df_benign.sample(n=min_count, random_state=42) df_malicious_balanced	=	df_malicious.sample(n=min_count, random_state=42)

# Combine and shuffle
df_balanced	=	pd.concat([df_benign_balanced, df_malicious_balanced]).sample(frac=1, random_state=42)

# Split into train and test (80-20)
train_df,	test_df	=	train_test_split(df_balanced,	test_size=0.2, stratify=df_balanced['Benign'], random_state=42)

# Save to CSV train_df.to_csv('balanced_train.csv', index=False) test_df.to_csv('balanced_test.csv', index=False)

print("Balanced	and	split	data	saved	as	'balanced_train.csv'	and 'balanced_test.csv'")
