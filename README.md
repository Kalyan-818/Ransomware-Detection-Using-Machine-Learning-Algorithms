# Ransomware-Detection-Using-Machine-Learning-Algorithms

SOURCE CODE

(i)train_model.py

import pandas as pd import joblib
from	sklearn.ensemble	import	GradientBoostingClassifier,
VotingClassifier
from sklearn.tree import DecisionTreeClassifier from sklearn.model_selection import train_test_split
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
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, stratify=y, random_state=42
)

# Lightweight Models
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
gb	=	GradientBoostingClassifier(n_estimators=50,	learning_rate=0.1, random_state=42)

# Voting Classifier (Lightweight Ensemble) voting_clf = VotingClassifier(estimators=[
("DecisionTree", dt), ("GradientBoosting", gb)
], voting='soft')

# Train ensemble voting_clf.fit(X_train, y_train)

# Evaluate
y_pred = voting_clf.predict(X_test)
print("\n✅	Lightweight	Ensemble	Model	Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("◻ Classification Report:\n", classification_report(y_test, y_pred)) # Save model

joblib.dump(voting_clf, "ransomware_model_lightweight.joblib") print("◻ Model saved as 'ransomware_model_lightweight.joblib'")

(ii)train_hash_model.py

import pandas as pd import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# Load the dataset
df = pd.read_csv('balanced_train.csv')

# Ensure correct column names
X = df.iloc[:, 1] # Second column is md5hash y = df.iloc[:, -1] # Last column is benign

# Encode hash strings to numbers encoder = LabelEncoder()
X_encoded = encoder.fit_transform(X)


# Train the model
model = RandomForestClassifier() model.fit(X_encoded.reshape(-1, 1), y)

# Save the model and encoder joblib.dump(model, 'hash_model.joblib') joblib.dump(encoder, 'hash_encoder.joblib')

print("✅ hash_model.joblib and hash_encoder.joblib saved successfully.")



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

(iv)index.html


<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Ransomware Detection Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1"
/>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min
.css" rel="stylesheet" />
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style> body {
font-family: 'Segoe UI', sans-serif; background-color: #121212;

color: #fff;
transition: background 0.5s, color 0.5s;
}
.card, .chart-container { border-radius: 10px; background-color: #1e1e1e;
box-shadow: 0 5px 15px rgba(0,0,0,0.3); padding: 15px;
margin-bottom: 20px; animation: fadeIn 0.8s ease;
}
.input-group input, .input-group button { border-radius: 10px;
}
.theme-toggle { position: absolute; top: 20px;
right: 20px;
}
.chart-container canvas { max-height: 200px;
}
.light-mode {
background-color: #f8f9fa !important; color: #000 !important;
}
.light-mode .card, .light-mode .chart-container, .light-mode .list-group- item {
background-color: #ffffff !important;

color: #000 !important;
}
@keyframes fadeIn {
from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); }
}
</style>
</head>
<body id="body" class="theme-dark">
<div class="container py-4">
<button	class="btn	btn-outline-light	theme-toggle" onclick="toggleTheme()">◻ / ☀</button>
<h3	class="text-center	mb-4">◻	Ransomware	Detection Dashboard</h3>

<!-- ◻ Hash Checker -->
<div class="card">
<form method="post" action="/hashcheck">
<div class="input-group">
<input	type="text"	name="hash_input"	class="form-control" placeholder="Enter MD5 hash..." required />
<button class="btn btn-secondary" type="submit">Check</button>
</div>
</form>
{% if hash_result %}
<div	class="mt-3	alert	alert-info	text-center">◻	Prediction:
<strong>{{ hash_result }}</strong></div>
{% endif %}
</div>

<!-- ◻ File Upload -->
<form method="post" enctype="multipart/form-data">
<div class="card">
<div class="input-group">
<input type="file" name="file" class="form-control" accept=".csv" required />
<button class="btn btn-primary" type="submit">Upload</button>
</div>
</div>
</form>

<!-- ◻ Best Model -->
{% if best_model_name %}
<div class="card text-center">
<h6>◻ Best Model</h6>
<h5 class="text-success">{{ best_model_name }}</h5>
<small	class="text-muted">Accuracy:
{{ best_model_score }}%</small>
</div>
{% endif %}


<!-- ◻ Prediction Results -->
{% if result %}
<div class="row text-center">
<div class="col-md-6">
<div class="card">
<small>Benign Count</small>
<h4>{{ result['Benign'] }}</h4>

</div>
</div>
<div class="col-md-6">
<div class="card">
<small>Malicious Count</small>
<h4>{{ result['Malicious'] }}</h4>
</div>
</div>
</div>

<!-- ◻ Accuracy & Confidence Visuals -->
<div class="row">
<div class="col-md-6">
<div class="chart-container">
<h6 class="text-center">◻ Malicious vs Benign (Pie)</h6>
<canvas id="pieChart"></canvas>
</div>
</div>
<div class="col-md-6">
<div class="chart-container">
<h6 class="text-center">◻ Confidence Score</h6>
<canvas id="barChart"></canvas>
</div>
</div>
</div>

<!-- ◻ Model Accuracy Chart -->
{% if model_scores %}
<div class="chart-container">

<h6 class="text-center">◻ Model Accuracy Comparison</h6>
<canvas id="modelChart"></canvas>
</div>
{% endif %}
{% endif %}


<!-- ◻ Top Predictions -->
{% if hash_predictions %}
<div class="card">
<h6>Top Hash Predictions</h6>
<ul class="list-group list-group-flush small">
{% for item in hash_predictions %}
<li class="list-group-item d-flex justify-content-between">
{{ item.hash }}
<span class="badge bg-{{ 'success' if item.prediction == 'Benign' else 'danger' }}">{{ item.prediction }}</span>
</li>
{% endfor %}
</ul>
</div>
{% endif %}
</div>

<!-- Theme Toggle Script -->
<script>
function toggleTheme() { document.getElementById('body').classList.toggle('light-mode');
}
</script>

<!-- Chart JS Scripts -->
<script>
{% if result %}
const resultData = {{ result | tojson }};
const modelScores = {{ model_scores | tojson if model_scores else 'null' }};
const bestModel = "{{ best_model_name }}";


new Chart(document.getElementById('pieChart'), { type: 'pie',
data: {
labels: ['Benign', 'Malicious'], datasets: [{
data: [resultData.Benign, resultData.Malicious], backgroundColor: ['#28a745', '#dc3545']
}]
}
});

new Chart(document.getElementById('barChart'), { type: 'bar',
data: {
labels: ['Benign', 'Malicious'], datasets: [{
label: 'Confidence', data: [
resultData.Benign / (resultData.Benign + resultData.Malicious), resultData.Malicious / (resultData.Benign + resultData.Malicious)

],
backgroundColor: ['#28a745', '#dc3545']
}]
},
options: {
scales: { y: { beginAtZero: true, max: 1 } }
}
});


if (modelScores) {
new Chart(document.getElementById('modelChart'), { type: 'bar',
data: {
labels: Object.keys(modelScores), datasets: [{
label: 'Accuracy (%)',
data: Object.values(modelScores),
backgroundColor: Object.keys(modelScores).map(name => name === bestModel ? '#28a745' : '#6c757d')
}]
},
options: {
scales: { y: { beginAtZero: true, max: 100 } }
}
});
}
{% endif %}
</script>
</body>

</html>



(v)app.py

from	flask	import	Flask,	render_template,	request,	flash,	redirect, send_file
import pandas as pd import joblib import os
import io
from werkzeug.utils import secure_filename

app = Flask(_name_) app.secret_key = 'supersecretkey' UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True) app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global vars
global_df = pd.DataFrame()
model_path = "ransomware_model_trained.joblib"
model = joblib.load(model_path) if os.path.exists(model_path) else None

@app.route('/', methods=['GET', 'POST']) def index():
global global_df
result, chart_data, model_scores, best_model_name, best_model_score, hash_predictions, hash_result = None, None, None, None, None, [], None

if request.method == 'POST': file = request.files.get('file')
if not file or not file.filename.endswith('.csv'): flash("Please upload a valid CSV file.", "danger") return redirect('/')

filename = secure_filename(file.filename)
path = os.path.join(app.config['UPLOAD_FOLDER'], filename) file.save(path)

try:
df = pd.read_csv(path) global_df = df.copy()

label_col = next((col for col in df.columns if col.lower() == 'benign'), None)
if label_col:
df.drop(columns=['FileName',	'md5Hash',	'mdhash'], errors='ignore', inplace=True)
X = df.drop(label_col, axis=1, errors='ignore') y_pred = model.predict(X)
df['predicted'] = y_pred

result = {
"Benign": int((y_pred == 1).sum()), "Malicious": int((y_pred == 0).sum())
}

chart_data = {
"Benign": result["Benign"], "Malicious": result["Malicious"]
}

hash_col = next((c for c in df.columns if c.lower() in ['md5hash', 'mdhash']), None)
if hash_col:
for _, row in df[[hash_col, 'predicted']].head(5).iterrows(): hash_predictions.append({
"hash": row[hash_col],
"prediction": "Benign" if row['predicted'] == 1 else
"Malicious"
})

except Exception as e: flash(f"Error: {e}", "danger")

return render_template("index.html", result=result, chart_data=chart_data, model_scores=model_scores,
best_model_name=best_model_name, best_model_score=best_model_score, hash_predictions=hash_predictions, hash_result=hash_result
)

@app.route('/hashcheck', methods=['POST'])

def hashcheck(): global global_df
hash_input = request.form.get('hash_input') hash_result = "Hash not found."

if not hash_input or global_df.empty: return redirect('/')

hash_col = next((c for c in global_df.columns if c.lower() in ['md5hash', 'mdhash']), None)
label_col = next((c for c in global_df.columns if c.lower() == 'benign'), None)

if hash_col and label_col:
match = global_df[global_df[hash_col] == hash_input] if not match.empty:
pred = match.iloc[0][label_col]
hash_result = "Benign" if pred == 1 else "Malicious"


return render_template("index.html", result=None,
chart_data=None, model_scores=None, best_model_name=None, best_model_score=None, hash_predictions=[], hash_result=hash_result
)

@app.route('/download', methods=['POST']) def download():
result_data = request.form.get('result_data') if not result_data:
flash("No data available to download.", "warning") return redirect('/')

try:
df	=	pd.DataFrame(eval(result_data).items(),	columns=["Label", "Count"])
buffer = io.StringIO() df.to_csv(buffer, index=False) buffer.seek(0)
return send_file(io.BytesIO(buffer.getvalue().encode()), mimetype="text/csv",
as_attachment=True, download_name="prediction_report.csv")
except Exception as e:
flash(f"Download error: {e}", "danger") return redirect('/')

if _name_ == "_main_": app.run(debug=True)
