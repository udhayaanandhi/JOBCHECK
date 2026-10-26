"""Retrain JobCheck with RandomOverSampler on TRAINING data only.

Run: python train_models.py
Optional: python train_models.py --skip-baseline
Backs up original models.joblib and metrics before replacing them.
"""
import argparse
import html
import os
import re
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
FIELDS = ['title', 'company_profile', 'description', 'requirements', 'benefits', 'location',
          'department', 'employment_type', 'required_experience', 'required_education',
          'industry', 'function']
SEED = 42


def clean_text(value):
    if pd.isna(value):
        return ''
    value = html.unescape(str(value))
    value = re.sub(r'<[^>]+>', ' ', value)
    value = re.sub(r'https?://\S+|www\.\S+', ' ', value)
    return re.sub(r'\s+', ' ', value).strip()


def build_models():
    return {
        'Multinomial Naive Bayes': MultinomialNB(alpha=0.5),
        'Logistic Regression': LogisticRegression(max_iter=1500, class_weight='balanced', random_state=SEED),
        'Linear SVM': LinearSVC(class_weight='balanced', random_state=SEED),
        'Decision Tree': DecisionTreeClassifier(max_depth=40, min_samples_leaf=2,
                                                class_weight='balanced', random_state=SEED),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'Hard Voting Classifier': VotingClassifier(estimators=[
            ('nb', MultinomialNB(alpha=0.5)),
            ('lr', LogisticRegression(max_iter=1500, class_weight='balanced', random_state=SEED)),
            ('svm', LinearSVC(class_weight='balanced', random_state=SEED))
        ], voting='hard')
    }


def train_group(Xtrain, ytrain, Xtest, ytest, knn_train, knn_test, label):
    models = build_models()
    records = []
    for name, model in models.items():
        xtr, xte = (knn_train, knn_test) if name == 'K-Nearest Neighbors' else (Xtrain, Xtest)
        start = time.perf_counter()
        model.fit(xtr, ytrain)
        elapsed = time.perf_counter() - start
        pred = model.predict(xte)
        score = None
        if hasattr(model, 'predict_proba'):
            score = model.predict_proba(xte)[:, 1]
        elif hasattr(model, 'decision_function'):
            score = model.decision_function(xte)
        tn, fp, fn, tp = confusion_matrix(ytest, pred, labels=[0, 1]).ravel()
        records.append({
            'Model': name, 'Accuracy': accuracy_score(ytest, pred),
            'Precision (fake=1)': precision_score(ytest, pred, zero_division=0),
            'Recall (fake=1)': recall_score(ytest, pred, zero_division=0),
            'F1 (fake=1)': f1_score(ytest, pred, zero_division=0),
            'ROC-AUC': roc_auc_score(ytest, score) if score is not None else np.nan,
            'PR-AUC': average_precision_score(ytest, score) if score is not None else np.nan,
            'Training Time (seconds)': round(elapsed, 3),
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
            'Training Approach': label,
        })
        print(f'  {name}: accuracy={records[-1]["Accuracy"]:.3f}, fake recall={records[-1]["Recall (fake=1)"]:.3f}, fake F1={records[-1]["F1 (fake=1)"]:.3f}', flush=True)
    return models, pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-baseline', action='store_true', help='Train oversampled models only')
    args = parser.parse_args()
    df = pd.read_excel(ROOT / 'data.xlsx')
    df = df.dropna(subset=['fraudulent']).copy()
    df['fraudulent'] = pd.to_numeric(df['fraudulent'], errors='coerce')
    df = df[df['fraudulent'].isin([0, 1])].copy()
    # Keep duplicates as originally supplied; do not change dataset size silently.
    texts = df[FIELDS].fillna('').astype(str).apply(
        lambda row: ' '.join(clean_text(v) for v in row), axis=1
    ).tolist()
    labels = df['fraudulent'].astype(int).to_numpy()
    idx = np.arange(len(labels))
    tr, te = train_test_split(idx, test_size=0.20, random_state=SEED, stratify=labels)
    ytrain, ytest = labels[tr], labels[te]
    print('Train class counts:', dict(zip(*np.unique(ytrain, return_counts=True))))
    print('Test class counts:', dict(zip(*np.unique(ytest, return_counts=True))))
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2, ngram_range=(1, 2),
                                 stop_words='english', sublinear_tf=True)
    Xtr = vectorizer.fit_transform([texts[i] for i in tr])
    Xte = vectorizer.transform([texts[i] for i in te])
    svd = TruncatedSVD(n_components=min(100, Xtr.shape[1]-1), random_state=SEED)
    scaler = StandardScaler()
    knn_tr = scaler.fit_transform(svd.fit_transform(Xtr))
    knn_te = scaler.transform(svd.transform(Xte))
    results_dir = ROOT / 'results'
    results_dir.mkdir(exist_ok=True)
    baseline = None
    if not args.skip_baseline:
        print('\nTraining original baseline models...', flush=True)
        _, baseline = train_group(Xtr, ytrain, Xte, ytest, knn_tr, knn_te, 'Original (imbalanced)')
        baseline.to_csv(results_dir / 'baseline_model_metrics.csv', index=False)
    print('\nOversampling TRAINING examples only...', flush=True)
    ros = RandomOverSampler(random_state=SEED)
    # Resample training row indices to keep sparse TF-IDF and KNN's SVD representations aligned.
    sampled_rows, y_bal = ros.fit_resample(np.arange(len(tr)).reshape(-1, 1), ytrain)
    sampled_rows = sampled_rows.ravel()
    X_bal = Xtr[sampled_rows]
    knn_bal = knn_tr[sampled_rows]
    print('Balanced training counts:', dict(zip(*np.unique(y_bal, return_counts=True))))
    print('Test set remains unchanged.', flush=True)
    print('\nTraining oversampled models...', flush=True)
    models, balanced = train_group(X_bal, y_bal, Xte, ytest, knn_bal, knn_te, 'Random oversampling')
    # Preserve existing artifacts in a dedicated backup folder.
    backup = ROOT / 'backup_before_oversampling'
    backup.mkdir(exist_ok=True)
    for file in [ROOT / 'models.joblib', results_dir / 'model_metrics.csv']:
        if file.exists() and not (backup / file.name).exists():
            shutil.copy2(file, backup / file.name)
    joblib.dump({'vectorizer': vectorizer, 'models': models, 'svd': svd,
                 'scaler': scaler, 'usable_rows': len(df),
                 'train_rows': len(tr), 'test_rows': len(te)}, ROOT / 'models.joblib')
    balanced.drop(columns=['Training Approach']).to_csv(results_dir / 'model_metrics.csv', index=False)
    balanced.to_csv(results_dir / 'oversampled_model_metrics.csv', index=False)
    if baseline is not None:
        comparison = pd.concat([baseline, balanced], ignore_index=True)
        comparison.to_csv(results_dir / 'before_after_comparison.csv', index=False)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(11, 5))
            names = baseline['Model'].tolist()
            pos = np.arange(len(names))
            width = .38
            ax.bar(pos-width/2, baseline['Recall (fake=1)'], width, label='Original')
            ax.bar(pos+width/2, balanced['Recall (fake=1)'], width, label='Oversampled')
            ax.set_xticks(pos, names, rotation=25, ha='right')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Fake-class recall')
            ax.set_title('Fraud detection before and after random oversampling')
            ax.legend()
            fig.tight_layout()
            fig.savefig(results_dir / 'oversampling_recall_comparison.png', dpi=160)
            plt.close(fig)
        except ImportError:
            print('matplotlib unavailable: skipped comparison plot')
    print('\nSaved models.joblib and results. Restart Flask to load new models.', flush=True)

if __name__ == '__main__':
    main()
