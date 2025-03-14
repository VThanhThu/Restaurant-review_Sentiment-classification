import os
import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud

# Định nghĩa đường dẫn
base_dir = r"D:\DS CLUB\Restaurent Review\Dataset"
eda_dir = os.path.join(base_dir, "EDA")

# Đọc dữ liệu
df = pd.read_csv(os.path.join(base_dir, "data.csv"))

# Loại bỏ khoảng trắng trong tên cột
df.columns = df.columns.str.strip()

# Vẽ biểu đồ số lượng phản hồi
plt.figure(figsize=(10, 8))
plt.title("Customer Feedback: Positive vs Negative")

# Vẽ countplot
plot = sns.countplot(x="Liked", hue="Liked", data=df, palette="Set2", legend=False)


# Thêm giá trị số lên trên thanh
for p in plot.patches:
    plot.annotate(f'{p.get_height()}', 
                  (p.get_x() + p.get_width() / 2, p.get_height() + 50), 
                  ha='center', va='center', 
                  fontsize=12, color='black')

# Lưu biểu đồ
plt.savefig(os.path.join(eda_dir, "restaurant_feedback_distribution.png"), dpi=300, bbox_inches="tight")
plt.close()

# Xử lý stop words
stop_words = set(text.ENGLISH_STOP_WORDS) 
words_to_keep = {
    "not", "never", "no", "none", "nor", "can not", "could not", "has not", "do",
    "always", "sometimes", "often", "rarely", "usually", "can", "has", "down",
    "many", "few", "most", "several", "some", "all", "twenty", 
    "good", "bad", "amazing", "terrible", "horrible", "awesome", "awful",
    "happy", "sad", "love", "hate", "great", "disgusting", "delicious", "too", "enough", 
    "should", "much", "become", "back", "go", "can", "less", "least",
    "take", "would", "again", "nothing", "everything", "get", "might", "util", "next",
    "couldn't", "hasn't", "over", "up", "must", "have"
}
stop_words = list(stop_words - words_to_keep) 

# Hàm làm sạch văn bản
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)  # Xóa HTML
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Xóa URL
        text = re.sub(r'[^a-z\s]', '', text)  # Xóa ký tự đặc biệt
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text
    return ""

# Áp dụng làm sạch dữ liệu
df["Clean_review"] = df["Review"].apply(clean_text)

# Chuyển đổi nhãn
label_encoder = LabelEncoder()
df["Label"] = label_encoder.fit_transform(df["Liked"])

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(df["Clean_review"])

# Lưu TF-IDF model
joblib.dump(tfidf_vectorizer, os.path.join(base_dir, "tfidf_vectorizer.pkl"))

# Tạo WordCloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(df["Clean_review"]))

# Lưu hình ảnh WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig(os.path.join(eda_dir, "restaurant_feedback_wordcloud.png"), dpi=300, bbox_inches="tight")
plt.close()

# Xóa cột không cần thiết
df.drop(["Review", "Liked"], axis=1, errors="ignore", inplace=True)

# Lưu dữ liệu đã xử lý
df.to_csv(os.path.join(base_dir, "data_training.csv"), index=False)

print("Done")
