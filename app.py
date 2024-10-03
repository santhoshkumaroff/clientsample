from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load datasets
df_standard = pd.read_csv('Standard File.csv')  # The standard file with descriptions and technical codes

# Function to find matching descriptions and percentages
def find_best_matches(new_description, df_standard, top_n=5):
    descriptions = df_standard['description'].tolist()
    technical_codes = df_standard['technical_code'].tolist()

    # Add the new description to the list
    descriptions.append(new_description)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer().fit_transform(descriptions)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(vectors[-1:], vectors[:-1]).flatten()

    # Create a DataFrame with technical codes and matching percentages
    similarity_df = pd.DataFrame({
        'description': df_standard['description'],
        'technical_code': df_standard['technical_code'],
        'similarity_percentage': cosine_similarities * 100  # Convert to percentage
    })

    # Sort by similarity percentage in descending order
    similarity_df = similarity_df.sort_values(by='similarity_percentage', ascending=False)

    # Return top N matches
    return similarity_df.head(top_n).to_dict(orient='records')

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# API route to handle AJAX requests
@app.route('/find_matches', methods=['POST'])
def find_matches():
    data = request.get_json()
    new_description = data.get('description')

    if new_description:
        matches = find_best_matches(new_description, df_standard)
        return jsonify(matches)

    return jsonify({'error': 'No description provided'})

if __name__ == '__main__':
    app.run(debug=True)
