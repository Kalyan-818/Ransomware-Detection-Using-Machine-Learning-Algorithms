import pandas as pd import joblib
from sklearn.ensemble import GradientBoostingClassifier,VotingClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("balanced_train.csv")

# Drop unnecessary columns
drop_cols = ["FileName", "md5Hash", "mdhash"]
df.drop(columns=[col	for	col	in	drop_cols	if	col	in	df.columns], inplace=True)

# Define target column
target_col = "Benign" if "Benign" in df.columns else "benign" print("◻ Label Distribution:\n", df[target_col].value_counts())

# Define features and labels

X = df.drop(columns=[target_col]) y = df[target_col]

# Split data for training
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, stratify=y, random_state=42)

# Lightweight Models
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
gb	=	GradientBoostingClassifier(n_estimators=50,	learning_rate=0.1, random_state=42)

# Voting Classifier (Lightweight Ensemble) voting_clf = VotingClassifier(estimators=[("DecisionTree", dt), ("GradientBoosting", gb)], voting='soft')

# Train ensemble voting_clf.fit(X_train, y_train)

# Evaluate
y_pred = voting_clf.predict(X_test)
print("\n✅	Lightweight	Ensemble	Model	Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("◻ Classification Report:\n", classification_report(y_test, y_pred)) # Save model

joblib.dump(voting_clf, "ransomware_model_lightweight.joblib") print("◻ Model saved as 'ransomware_model_lightweight.joblib'")
