import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report

# Đọc dữ liệu
df = pd.read_csv("D:\\DS CLUB\\Restaurent Review\\Dataset\\training_data.csv")

# Load TF-IDF đã lưu
tfidf_vectorizer = joblib.load("D:\\DS CLUB\\Restaurent Review\\Dataset\\tfidf_vectorizer.pkl")

# Chuyển đổi văn bản thành vector TF-IDF
df["Clean_review"] = df["Clean_review"].fillna("")  # Xử lý NaN
X = tfidf_vectorizer.transform(df["Clean_review"]).toarray().astype('float32')
y = df["Label"].values.astype('float32')

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Huấn luyện Naive Bayes
nb_clf = BernoulliNB()
nb_clf.fit(X_train, y_train)

# Dự đoán trên tập test
test_ds_predicted = nb_clf.predict(X_test)

# Lưu classification report vào file .txt
report = classification_report(y_test, test_ds_predicted)
report_path = "D:\\DS CLUB\\Restaurent Review\\Dataset\\result_NBayes\\classification_report.txt"

with open(report_path, "w") as f:
    f.write(report)

print(f"Classification report saved to {report_path}")

# Tạo confusion matrix
cm = confusion_matrix(y_test, test_ds_predicted)

# Vẽ heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)

# Đặt nhãn
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

# Lưu file ảnh
conf_matrix_path = "D:\\DS CLUB\\Restaurent Review\\Dataset\\result_NBayes\\confusion_matrix.png"
plt.savefig(conf_matrix_path, dpi=300, bbox_inches="tight")
plt.close()  

print(f"Confusion matrix saved to {conf_matrix_path}")

# Lưu kết quả dự đoán vào CSV
results = pd.DataFrame({
    'True Labels': y_test,  
    'Predicted Labels': test_ds_predicted  
})

results_path = "D:\\DS CLUB\\Restaurent Review\\Dataset\\result_NBayes\\predictions.csv"
results.to_csv(results_path, index=False)

print(f"Predictions saved to {results_path}")
