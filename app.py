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
