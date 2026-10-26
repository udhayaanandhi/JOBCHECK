
# JobCheck — Installation and User Guide

## 1. Project Overview

JobCheck is a Flask-based web application that predicts whether a job posting is Genuine or Fraudulent using six machine learning classifiers.

The application provides job checking, dataset analysis, model performance visualizations, scan history, report downloads, and feedback functionality.

## 2. Before You Start

1. Back up your existing JobCheck project folder.
2. Preserve your existing `models.joblib`, `data.xlsx`, `results/`, `jobcheck_history.sqlite3`, uploads, and virtual environment.
3. Extract the updated project files into an appropriate folder.
4. If you are updating an existing project, replace only the files specified for that update.
5. Review any custom routes or helper functions before replacing `app.py`.

## 3. Installation on Windows

### Step 1: Open the Project

Open the JobCheck project folder in VS Code or Command Prompt.

### Step 2: Activate the Virtual Environment

If your project already has a virtual environment named `venv`, run:

```bat
venv\Scripts\activate
```

### Step 3: Install Dependencies

Run:

```bat
python -m pip install -r requirements.txt
```

### Step 4: Start the Application

Run:

```bat
python app.py
```

### Step 5: Open JobCheck

Open your browser and visit:

http://127.0.0.1:5000

Keep the terminal running while you use the application.

## 4. Application Pages and Features

### Home — `/`

- Enter job posting details for prediction.
- View the predicted Genuine or Fraudulent result.
- Review warning phrases and an employer-checklist summary.
- View model disagreement information.
- Download prediction reports and submit feedback.

### Dataset Analysis — `/dataset`

- Upload a CSV or Excel file.
- Analyze job postings using the available prediction models.
- View prediction distribution charts.
- Review model metrics and confusion matrices when a valid `fraudulent` column containing 0/1 labels is available.
- Export results to CSV.

### Model Performance — `/performance`

- View metrics loaded from `results/model_metrics.csv`.
- Review available confusion-matrix images.

### Scan History — `/history`

- Review previously saved scans.
- Download available reports.
- View or submit feedback.

### Health Check — `/health`

Provides the application's health-check endpoint.

## 5. Machine Learning Models

JobCheck uses six classifiers:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Machine (Linear SVM)
4. Decision Tree
5. K-Nearest Neighbors (KNN)
6. Hard Voting Classifier

The Hard Voting Classifier combines the predictions of Naive Bayes, Logistic Regression, and Linear SVM.

The existing trained models are stored in `models.joblib`. Whether the models are retrained depends on the training script and the project version being used.

### Prediction Labels

- `0` — Genuine job posting
- `1` — Fraudulent job posting

Predictions are estimates based on patterns learned from the training dataset. They do not independently verify an employer's identity or guarantee that a job is legitimate.

## 6. Dataset Analysis Notes

- Use job postings that were not used during model training when evaluating generalization.
- Evaluating the models on their training data can produce misleadingly high scores.
- A valid `fraudulent` column with 0/1 labels is needed for comparison against known outcomes.
- Large datasets containing 15,000 or more records may take several minutes to process, particularly when KNN is used.

## 7. Scan History and Feedback

The application uses a local SQLite database named `jobcheck_history.sqlite3` to store supported history and feedback information.

Some application versions use the tables `scans_v2` and `feedback_v2`. In those versions, existing tables remain untouched, but old records may not appear on the updated History page.

Feedback is stored for review and does not automatically retrain the machine learning models.

When moving to a fresh project folder, remember that a newly created database will not contain your previous scan history.

## 8. Updating an Existing Project Safely

1. Back up the entire project folder.
2. Identify the files included in the update package.
3. Replace only the intended files.
4. Preserve `models.joblib`, `data.xlsx`, `results/`, the SQLite database, uploads, and the virtual environment unless the update explicitly requires replacements.
5. If `app.py` is replaced, check for custom routes and helper functions that must be retained.
6. Restart the application and verify that the Home, Dataset, Performance, and History pages work correctly.

Do not replace your entire existing project with a partial update package.

## 9. Windows Quick Start

For an existing project with a virtual environment:

```bat
cd path\to\JobCheck
venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

Then open:

http://127.0.0.1:5000

Replace `path\to\JobCheck` with the actual location of your project folder.

## 10. Troubleshooting

### Application Does Not Start

- Activate the correct virtual environment.
- Install the dependencies listed in `requirements.txt`.
- Check the terminal for Python errors.
- Confirm that the required model and dataset files exist.

### Changes Are Not Visible

- Stop and restart Flask.
- Hard-refresh the browser using `Ctrl + F5`.

### Scan History Is Missing

- Confirm that the application is using the original `jobcheck_history.sqlite3`.
- Check whether the updated version uses different database tables.
- Avoid creating a fresh database if you need to preserve previous scans.

### Custom Routes Have Disappeared

Restore the backup and merge the custom routes and helper functions into the updated `app.py`.

### Dataset Processing Takes a Long Time

Processing large datasets can take several minutes. KNN may be particularly slow because it performs distance-based comparisons.

## 11. Project Safety

- Keep backups of your project files and local database.
- Do not overwrite trained models unless you intend to replace them.
- Do not upload sensitive information in job descriptions or datasets.
- Treat model predictions as decision-support information, not definitive proof of fraud.
- Verify important employment details independently before accepting a job offer.
