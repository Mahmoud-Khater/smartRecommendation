from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 🧠 نخزن المقالات في قائمة داخل الميموري
articles = []

# 🔧 Endpoint لإضافة مقال جديد
@app.route('/articles', methods=['POST'])
def add_article():
    data = request.get_json()
    article_id = len(articles) + 1  # ID تلقائي
    article = {
        "id": article_id,
        "title": data['title'],
        "content": data['content']
    }
    articles.append(article)
    return jsonify({"message": "Article added", "article": article}), 201

# 🔍 Endpoint للتوصية
@app.route('/recommend', methods=['POST'])
def recommend():
    if not articles:
        return jsonify({"message": "No articles available for comparison"}), 400

    data = request.get_json()
    input_text = data['content']

    # كل المقالات الحالية + المقال الجديد
    all_texts = [input_text] + [a['content'] for a in articles]

    # نحسب الـ TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # نحسب التشابه
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # نختار أعلى 3 تشابهات
    top_indices = similarities.argsort()[::-1][:3]
    recommended = [articles[i] for i in top_indices]

    return jsonify(recommended)

# 📖 Endpoint لاستعراض كل المقالات
@app.route('/articles', methods=['GET'])
def get_articles():
    return jsonify(articles)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
