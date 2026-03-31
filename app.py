from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv("netflix_titles.csv")

# Fill missing values
df['description'] = df['description'].fillna('')
df['listed_in'] = df['listed_in'].fillna('')

# Combine features
df['content'] = df['description'] + " " + df['listed_in']

# Convert text → vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])

# Compute similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Reset index
df = df.reset_index()

# Function to recommend movies
def recommend(title):
    try:
        idx = df[df['title'].str.lower() == title.lower()].index[0]
        scores = list(enumerate(cosine_sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

        movie_indices = [i[0] for i in scores]
        return df['title'].iloc[movie_indices].tolist()
    except:
        return ["Movie not found 😅 Try another one!"]

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]

    if "hi" in user_input.lower():
        return jsonify({"reply": "Hey 🎬! Tell me a movie you like, I'll recommend similar ones!"})

    else:
        recommendations = recommend(user_input)
        return jsonify({"reply": "🎥 Recommendations:\n" + "\n".join(recommendations)})

if __name__ == "__main__":
    app.run(debug=True)