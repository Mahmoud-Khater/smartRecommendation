from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
import nltk
import re
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')  # خفيف وسريع ودقيق
tokenizer = T5Tokenizer.from_pretrained("t5-base")
title_model = T5ForConditionalGeneration.from_pretrained("t5-base")
emotion_model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)



app = Flask(__name__)

# db
db = mysql.connector.connect(
    host="localhost",
    user="root",       # ✏️ غيّرها حسب جهازك
    password="1234",   # ✏️ غيّرها
    database="recommender"
)

# clean data
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # 4. Join the words back
    cleaned_text = ' '.join(words)
    
    return cleaned_text

def generate_title_from_content(content):
    input_text = f"summarize: {content}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = title_model.generate(inputs, max_length=12, min_length=4, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def classify_article(content):
    inputs = emotion_tokenizer(content, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    label = emotion_model.config.id2label[predicted_class_id]
    return label


cursor = db.cursor(dictionary=True)

# 🧠 نخزن المقالات في قائمة داخل الميموري
articles = []


import json

@app.route('/generate-title', methods=['POST'])
def generate_title():
    data = request.get_json()
    content = data['content']

    # 👇 هنستخدم موديل ذكي يولد عنوان
    title = generate_title_from_content(content)

    return jsonify({"generated_title": title})


# 🔧 Endpoint لإضافة مقال جديد
@app.route('/articles', methods=['POST'])
def add_article():
    data = request.get_json()
    title = data.get('title')
    content = data['content']

    # توليد embedding
    embedding = model.encode(content).tolist()
    embedding_json = json.dumps(embedding)

    # توليد عنوان لو مش موجود
    if not title:
        title = generate_title_from_content(content)

    # تصنيف المقال
    topic = classify_article(content)

    sql = "INSERT INTO articles (title, content, embedding, topic) VALUES (%s, %s, %s, %s)"
    val = (title, content, embedding_json, topic)
    cursor.execute(sql, val)
    db.commit()

    return jsonify({"message": "Article added successfully!", "topic": topic})





# 🔍 Endpoint للتوصية
@app.route('/recommend', methods=['POST'])
def recommend():
    cursor.execute("SELECT * FROM articles")
    articles = cursor.fetchall()

    if not articles:
        return jsonify({"message": "No articles in database"}), 400

    data = request.get_json()
    input_text = data['content']

    # 🧠 BERT vector للمحتوى المطلوب
    input_vector = model.encode(input_text)

    # 🧠 نحول كل embedding من JSON string → numpy array
    article_vectors = [np.array(json.loads(a['embedding'])) for a in articles]

    similarities = cosine_similarity([input_vector], article_vectors).flatten()
    top_indices = similarities.argsort()[::-1][:3]
    recommended = [articles[i] for i in top_indices]

    return jsonify(recommended)




# 📖 Endpoint لاستعراض كل المقالات
@app.route('/articles', methods=['GET'])
def get_articles():
    cursor.execute("SELECT * FROM articles")
    articles = cursor.fetchall()
    return jsonify(articles)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
