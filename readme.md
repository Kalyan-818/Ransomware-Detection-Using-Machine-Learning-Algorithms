Ransomware Detection Using Machine Learning Algorithms

A Machine Learning-based cybersecurity application designed to identify potentially malicious files and distinguish between **Benign and Malicious activities** using classification algorithms. The project combines data preprocessing, machine learning model training, hash-based lookup, and an interactive Flask web dashboard.

Project Overview

Ransomware is a type of malware that restricts access to a victim's files or systems, typically by encrypting data and demanding payment for its release.

This project explores the use of Machine Learning techniques to identify suspicious and potentially malicious file records based on their extracted features.

The system uses classification algorithms to analyze input data and predict whether a record belongs to the Benign or Malicious category. A Flask-based web interface allows users to upload CSV datasets, view prediction results, and inspect hash information.

Project Objectives

- Develop a Machine Learning-based ransomware detection system.
- Classify file records into Benign and Malicious categories.
- Apply data preprocessing and class balancing techniques.
- Train and evaluate classification models.
- Explore hash-based identification of file records.
- Develop an interactive web dashboard for predictions and visualization.
- Present detection results in an understandable format.

Technologies Used

| Category | Technologies |

| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Algorithms | Random Forest, Decision Tree, Gradient Boosting |
| Data Processing | Pandas, NumPy |
| Model Serialization | Joblib |
| Backend | Flask |
| Frontend | HTML, CSS, Bootstrap 5 |
| Visualization | Chart.js |
| Hash Information | MD5 hash fields |

Machine Learning Algorithms

The project explores the following classification techniques:

1. Decision Tree Classifier

A supervised machine learning algorithm that makes predictions using a tree-like structure of decision rules.

2. Gradient Boosting Classifier

An ensemble learning algorithm that builds multiple weak prediction models sequentially to improve classification performance.

3. Voting Classifier

A soft-voting ensemble combines the probability predictions of Decision Tree and Gradient Boosting classifiers to produce a final classification.

4. Random Forest Classifier

An ensemble algorithm that combines multiple decision trees to perform classification. It is also explored separately in the hash-based model training script.

Project Workflow

1. Dataset Collection: Load the ransomware-related CSV dataset.
2. Data Preprocessing: Identify relevant features and remove unnecessary columns.
3. Class Balancing: Balance Benign and Malicious records using undersampling.
4. Dataset Splitting: Divide the balanced dataset into 80% training and 20% testing data.
5. Model Training:** Train classification models using Scikit-learn.
6. Model Evaluation: Evaluate predictions using accuracy and classification reports.
7. Model Serialization: Save trained models using Joblib.
8. Web Integration: Integrate the trained model with a Flask application.
9. Prediction: Upload CSV records and classify them as Benign or Malicious.
10. Visualization: Display classification counts and charts on the dashboard.

Project Architecture


Ransomware Detection System
│
├── Dataset
│   └── data_file.csv
│
├── Data Preprocessing
│   └── train_test_split.py
│
├── Model Training
│   ├── train_model.py
│   └── train_hash_model.py
│
├── Backend
│   └── app.py
│
├── Frontend
│   └── templates/
│       └── index.html
│
├── Generated Files
│   ├── balanced_train.csv
│   ├── balanced_test.csv
│   ├── ransomware_model_lightweight.joblib
│   ├── hash_model.joblib
│   └── hash_encoder.joblib
│
└── README.md


 File Descriptions

| File | Description |
|---|---|
| `app.py` | Flask backend responsible for CSV uploads, predictions, hash lookup, and report downloads. |
| `data_file.csv` | Original dataset containing file-related features and classification labels. |
| `index.html` | Frontend dashboard for interacting with the detection system. |
| `train_test_split.py` | Performs class balancing and splits the dataset into training and testing sets. |
| `train_model.py` | Trains a Decision Tree and Gradient Boosting soft-voting ensemble. |
| `train_hash_model.py` | Trains a Random Forest model using encoded hash values. |




Application Features

1. CSV Dataset Upload

Upload a CSV file containing file-related features. The application processes the records and generates model predictions.

2. Benign and Malicious Classification

The trained classification model predicts whether each supported input record is Benign or Malicious.

3. Hash-Based Lookup

Users can enter an MD5 hash to search for a matching record in the uploaded dataset and retrieve its existing classification label.

4. Interactive Dashboard

The web interface provides:

- Benign record count.
- Malicious record count.
- Pie chart visualization.
- Prediction distribution chart.
- Hash prediction information.
- Dark and light theme support.

5. Prediction Report

The application includes functionality to export prediction summary information in CSV format.

Model Evaluation

The training implementation uses the following evaluation metrics:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Classification Report

These metrics help assess the classification performance of the trained models.

Note: Actual model performance values should be added after running the training process on the dataset. No unverified accuracy figures are reported here.

Security Considerations

- The application is intended for educational and research purposes.
- CSV files should be validated before processing.
- Uploaded files should be handled securely.
- Model predictions should not be treated as a substitute for a dedicated endpoint protection solution.
- Hash matching only identifies records already present in the uploaded dataset; it does not independently establish whether a file is malicious.

Future Enhancements

- Integrate real-time file scanning.
- Add static and dynamic malware analysis.
- Explore advanced ML and Deep Learning models.
- Integrate SHA-256 file hashing and a verified hash database.
- Improve feature engineering and model evaluation.
- Add model comparison and performance dashboards.
- Implement secure authentication and user management.
- Deploy the application as a scalable cybersecurity monitoring service.

Project Information

Project Title: Ransomware Detection Using Machine Learning Algorithms

Domain: Cybersecurity | Machine Learning | Artificial Intelligence

Development Approach: Supervised Machine Learning with Flask Web Integration

