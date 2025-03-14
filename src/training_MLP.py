import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix

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

# Xây dựng mô hình MLP
model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(1, activation='sigmoid')  
])

# Compile mô hình
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# EarlyStopping + ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                    validation_data=(X_test, y_test), 
                    callbacks=[early_stopping, reduce_lr], verbose=1)

# Kiểm tra kết quả huấn luyện
print(f"Best Validation Loss: {min(history.history['val_loss'])}")
print(f"Best Validation Accuracy: {max(history.history['val_accuracy'])}")

# Lưu mô hình để tái sử dụng
model.save("D:\\DS CLUB\\Restaurent Review\\Dataset\\result\\sentiment_model.h5")

# Vẽ biểu đồ loss và accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over epochs')

plt.savefig("D:\\DS CLUB\\Restaurent Review\\Dataset\\result\\training_performance.png", dpi=300, bbox_inches="tight")
plt.close()

# Dự đoán nhãn test
y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int)

# Ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred_labels)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.savefig("D:\\DS CLUB\\Restaurent Review\\Dataset\\result\\confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# Lưu kết quả dự đoán vào file
output_path = "D:\\DS CLUB\\Restaurent Review\\Dataset\\result\\predictions.txt"
with open(output_path, "w") as f:
    for i in range(len(y_test)):
        f.write(f"Review {i+1}: Predicted Label = {y_pred_labels[i][0]}, True Label = {y_test[i]}\n")

print(f"Predictions saved to {output_path}")
