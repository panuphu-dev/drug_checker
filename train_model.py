"""
ML Model Training for Online Drug Sales Detection
โปรเจค: แนวทางการแก้ไขปัญหาการขายยาออนไลน์
มหาวิทยาลัยขอนแก่น - คณะเภสัชศาสตร์
"""

import pandas as pd
import numpy as np
import json
import joblib
import re
import os
import warnings
warnings.filterwarnings('ignore')

from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ====== 1. LOAD & CLEAN DATA ======
print("=" * 60)
print("📊 STEP 1: Loading and Cleaning Data")
print("=" * 60)

df = pd.read_excel('/mnt/project/แบบฟอร_มเก_บข_อม_ล_share_สำเนา.xlsx', sheet_name='ข้อมูลตัวอย่าง')
df = df.iloc[1:]  # Skip template row

print(f"Total samples: {len(df)}")
print(f"Label distribution:\n{df['สรุปผล(คนเดียว)'].value_counts()}")

# Create binary label
df['label'] = df['สรุปผล(คนเดียว)'].map({'ถูกกฎหมาย': 0, 'ผิดกฎหมาย': 1})
df = df[df['label'].notna()].copy()

# Combine text features
df['ชื่อสินค้า'] = df['ชื่อสินค้า'].fillna('')
df['ข้อความโฆษณา'] = df['ข้อความโฆษณา'].fillna('')
df['text'] = df['ชื่อสินค้า'] + ' ' + df['ข้อความโฆษณา']

# Clean text
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text_clean'] = df['text'].apply(clean_text)

# ====== 2. THAI NLP TOKENIZATION ======
print("\n" + "=" * 60)
print("🔤 STEP 2: Thai NLP Tokenization")
print("=" * 60)

stop_words = set(thai_stopwords())
custom_stops = {'ที่', 'ใน', 'ของ', 'และ', 'เป็น', 'มี', 'ได้', 'ไม่', 'จะ', 'กับ',
                'ให้', 'แต่', 'ก็', 'หรือ', 'ว่า', 'ถ้า', 'จาก', 'โดย', 'เมื่อ',
                'แล้ว', 'ยัง', 'คือ', 'ด้วย', 'ไป', 'มา', 'ทำ', 'อยู่', 'เรา', 'นี้',
                'ซึ่ง', 'กัน', 'ๆ', 'nan', 'NaN', 'None', ''}
stop_words.update(custom_stops)

def tokenize_thai(text):
    tokens = word_tokenize(str(text), engine='newmm')
    tokens = [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1 and t not in stop_words]
    return ' '.join(tokens)

print("Tokenizing Thai text...")
df['text_tokenized'] = df['text_clean'].apply(tokenize_thai)
print(f"Sample tokenized: {df['text_tokenized'].iloc[1][:100]}...")

# ====== 3. FEATURE ENGINEERING ======
print("\n" + "=" * 60)
print("⚙️ STEP 3: Feature Engineering")
print("=" * 60)

# Additional features
df['has_registration'] = df['เลขทะเบียน อย.'].notna().astype(int)
df['has_ad_license'] = df['เลขอนุญาตโฆษณา'].notna().astype(int)
df['text_length'] = df['text_clean'].str.len()
df['word_count'] = df['text_tokenized'].str.split().str.len()

# Keyword-based features
danger_keywords = ['อันตราย', 'ยาอันตราย', 'ยาควบคุม', 'สเตียรอยด์', 'steroid',
                   'ยาปฏิชีวนะ', 'ยาแก้อักเสบ', 'antibiotic', 'ยาฆ่าเชื้อ']
claim_keywords = ['รักษา', 'หาย', 'ป้องกัน', 'ต้าน', 'กำจัด', 'ฆ่า',
                  'มะเร็ง', 'เบาหวาน', 'ความดัน', 'หัวใจ', 'ไต', 'ตับ',
                  'รับรอง', 'การันตี', 'ปาฏิหาริย์', '100%', 'ชัวร์']
supplement_keywords = ['อาหารเสริม', 'ผลิตภัณฑ์เสริมอาหาร', 'สมุนไพร',
                       'ธรรมชาติ', 'สกัด', 'extract']

df['has_danger_keywords'] = df['text_clean'].apply(
    lambda x: int(any(k in str(x) for k in danger_keywords)))
df['has_claim_keywords'] = df['text_clean'].apply(
    lambda x: int(any(k in str(x) for k in claim_keywords)))
df['has_supplement_keywords'] = df['text_clean'].apply(
    lambda x: int(any(k in str(x) for k in supplement_keywords)))

print(f"Features created: registration, ad_license, text_length, word_count, keyword features")

# ====== 4. PREPARE DATA ======
X_text = df['text_tokenized'].values
y = df['label'].astype(int).values

print(f"\nDataset: {len(X_text)} samples")
print(f"Class 0 (ถูกกฎหมาย): {sum(y==0)}")
print(f"Class 1 (ผิดกฎหมาย): {sum(y==1)}")
print(f"Imbalance ratio: 1:{sum(y==0)//max(sum(y==1),1)}")

# ====== 5. TRAIN MULTIPLE MODELS ======
print("\n" + "=" * 60)
print("🤖 STEP 4: Training Multiple ML Models with Cross-Validation")
print("=" * 60)

# TF-IDF Vectorizer (use default tokenizer on pre-joined tokens)
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    token_pattern=r'(?u)\b\w+\b'
)

# Transform text
X_tfidf = tfidf.fit_transform(X_text)
print(f"TF-IDF features: {X_tfidf.shape[1]}")

# Apply SMOTE for oversampling minority class
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)
print(f"After SMOTE: {X_resampled.shape[0]} samples")
print(f"  Class 0: {sum(y_resampled==0)}, Class 1: {sum(y_resampled==1)}")

# Models to evaluate
models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=1000, C=1.0, random_state=42
    ),
    'SVM (RBF)': SVC(
        class_weight='balanced', kernel='rbf', probability=True, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        class_weight='balanced', n_estimators=200, max_depth=20, random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    ),
    'Naive Bayes': MultinomialNB(alpha=0.5)
}

# Manual cross-validation with SMOTE inside each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    print(f"\n{'─'*40}")
    print(f"🔹 Training: {name}")
    
    y_pred_all = np.zeros(len(y))
    y_proba_all = np.zeros(len(y))
    
    for train_idx, test_idx in cv.split(X_tfidf, y):
        X_train, X_test = X_tfidf[train_idx], X_tfidf[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Apply SMOTE only on training data
        try:
            smote_cv = SMOTE(random_state=42, k_neighbors=min(3, sum(y_train==1)-1))
            X_train_sm, y_train_sm = smote_cv.fit_resample(X_train, y_train)
        except:
            X_train_sm, y_train_sm = X_train, y_train
        
        model.fit(X_train_sm, y_train_sm)
        y_pred_all[test_idx] = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_proba_all[test_idx] = model.predict_proba(X_test)[:, 1]
    
    y_pred = y_pred_all.astype(int)
    
    try:
        auc = roc_auc_score(y, y_proba_all)
    except:
        auc = 0.0
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    spec = recall_score(y, y_pred, pos_label=0)
    cm = confusion_matrix(y, y_pred)
    
    results[name] = {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall_sensitivity': round(rec, 4),
        'specificity': round(spec, 4),
        'f1_score': round(f1, 4),
        'auc_roc': round(auc, 4),
        'confusion_matrix': cm.tolist()
    }
    
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Sensitivity: {rec:.4f}")
    print(f"  Specificity: {spec:.4f}")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  Confusion Matrix:\n  {cm}")

# ====== 6. SELECT BEST MODEL & RETRAIN ======
print("\n" + "=" * 60)
print("🏆 STEP 5: Select Best Model")
print("=" * 60)

# Select best by F1 score (important for imbalanced data)
best_name = max(results, key=lambda k: results[k]['f1_score'])
print(f"Best model by F1-Score: {best_name}")
print(f"  F1: {results[best_name]['f1_score']}")
print(f"  Sensitivity: {results[best_name]['recall_sensitivity']}")
print(f"  Specificity: {results[best_name]['specificity']}")

# Retrain best model on full SMOTE'd data
best_model = models[best_name]
smote_final = SMOTE(random_state=42, k_neighbors=3)
X_final, y_final = smote_final.fit_resample(X_tfidf, y)
best_model.fit(X_final, y_final)

# ====== 7. SAVE MODEL & ARTIFACTS ======
print("\n" + "=" * 60)
print("💾 STEP 6: Saving Model & Artifacts")
print("=" * 60)

output_dir = '/home/claude/model_output'
os.makedirs(output_dir, exist_ok=True)

# Save model
joblib.dump(best_model, f'{output_dir}/best_model.joblib')
joblib.dump(tfidf, f'{output_dir}/tfidf_vectorizer.joblib')

# Save results
with open(f'{output_dir}/evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# Save feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    feature_names = tfidf.get_feature_names_out()
    importances = best_model.feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:30]
    with open(f'{output_dir}/top_features.json', 'w', encoding='utf-8') as f:
        json.dump([{'feature': f, 'importance': round(float(i), 6)} for f, i in top_features],
                  f, ensure_ascii=False, indent=2)
elif hasattr(best_model, 'coef_'):
    feature_names = tfidf.get_feature_names_out()
    importances = np.abs(best_model.coef_[0]) if best_model.coef_.ndim > 1 else np.abs(best_model.coef_)
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:30]
    with open(f'{output_dir}/top_features.json', 'w', encoding='utf-8') as f:
        json.dump([{'feature': f, 'importance': round(float(i), 6)} for f, i in top_features],
                  f, ensure_ascii=False, indent=2)

# Save model info
model_info = {
    'best_model': best_name,
    'n_samples': len(y),
    'n_legal': int(sum(y == 0)),
    'n_illegal': int(sum(y == 1)),
    'n_features': X_tfidf.shape[1],
    'cv_folds': 5,
    'all_results': results
}
with open(f'{output_dir}/model_info.json', 'w', encoding='utf-8') as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)

# Save keyword lists for web app
keywords_config = {
    'danger_keywords': danger_keywords,
    'claim_keywords': claim_keywords,
    'supplement_keywords': supplement_keywords
}
with open(f'{output_dir}/keywords_config.json', 'w', encoding='utf-8') as f:
    json.dump(keywords_config, f, ensure_ascii=False, indent=2)

print(f"✅ Model saved to: {output_dir}/")
print(f"✅ Best model: {best_name}")

# ====== 8. SUMMARY TABLE ======
print("\n" + "=" * 60)
print("📋 FINAL RESULTS SUMMARY")
print("=" * 60)

print(f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Sensitivity':>12} {'Specificity':>12} {'F1-Score':>10} {'AUC-ROC':>10}")
print("─" * 95)
for name, r in results.items():
    marker = " ⭐" if name == best_name else ""
    print(f"{name:<25} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall_sensitivity']:>12.4f} {r['specificity']:>12.4f} {r['f1_score']:>10.4f} {r['auc_roc']:>10.4f}{marker}")

print(f"\n⭐ = Best model selected: {best_name}")
print("\n✅ Training complete!")
