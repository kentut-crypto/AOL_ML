import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import requests

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

MODEL_FILES = {
    'processed_recipes.csv': '1_2yKUpftAkQwlpelxh2jryyomSsRhWRb',
    'tfidf_vectorizer.joblib': '1lCSxkKvmrAqPMuh5n-JYft5usaxEVHR6',
    'ingredient_vectors.joblib': '18pjbExrvpLJCtTongyRnX_5kH3waF1ZA',
}

for filename, file_id in MODEL_FILES.items():
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename} from Google Drive..."):
            download_file_from_google_drive(file_id, filename)

class RecipeRecommender:
    def __init__(self):
        self.df = pd.read_csv('processed_recipes.csv')
        self.vectorizer = joblib.load('tfidf_vectorizer.joblib')
        self.ingredient_vectors = joblib.load('ingredient_vectors.joblib')

    def recommend_recipes(self, input_ingredients, top_n=5):
        if not input_ingredients:
            return []

        input_ingredient_str = ' '.join(input_ingredients)
        input_vector = self.vectorizer.transform([input_ingredient_str])
        similarities = cosine_similarity(input_vector, self.ingredient_vectors)[0]

        top_indices = similarities.argsort()[-top_n:][::-1]

        recommendations = []
        for idx in top_indices:
            try:
                recommendations.append({
                    'recipe_name': self.df.iloc[idx, 0],  # Recipe name
                    'ingredients': eval(self.df.iloc[idx, 10]) if isinstance(self.df.iloc[idx, 10], str) else self.df.iloc[idx, 10],
                    'ingredients_list': self.df.iloc[idx, 10],
                    'similarity_score': similarities[idx],
                    'nutrition_values': self.df.iloc[idx, 6],
                    'steps': self.df.iloc[idx, 8]
                })
            except Exception as e:
                st.warning(f"Could not process recipe: {e}")

        return recommendations

    def ingredient_overlap(self, input_ingredients, recipe_ingredients):
        input_set = set(input_ingredients)
        recipe_set = set(recipe_ingredients)

        overlap = len(input_set.intersection(recipe_set))
        return (overlap / len(input_set)) * 100 if input_set else 0

def main():
    # Set page title and layout
    st.set_page_config(page_title="Recipe Recommender", page_icon="🍽️", layout="wide")

    # Loading placeholder
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown("<h1 style='text-align: center; color: grey;'>🍽️</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: grey;'>Loading Delicious Recipes...</h3>", unsafe_allow_html=True)

    # Initialize recommender
    try:
        recommender = RecipeRecommender()
    except FileNotFoundError:
        loading_placeholder.error("Recipe dataset not found. Please check the file path.")
        st.stop()

    # Clear loading placeholder after successful load
    loading_placeholder.empty()

    # Main title
    st.title("🍳 Smart Recipe Recommender")

    # Sidebar instructions
    st.sidebar.title("How to Use")
    st.sidebar.info(
        "1. Enter ingredients you want to use\n"
        "2. Separate ingredients with commas\n"
        "3. Press Enter or click 'Recommend Recipes'\n"
        "Example: chicken, onion, garlic"
    )

    # Ingredients input with key to track changes
    ingredients_input = st.text_input(
        "Enter your ingredients (comma-separated)",
        placeholder="e.g. chicken, rice, bell pepper",
        key="ingredients_key"
    )

    # Number of recommendations slider
    num_recommendations = st.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=10,
        value=5
    )

    # Button to trigger recommendations
    recommend_clicked = st.button("Recommend Recipes")

    # Show recommendations if button clicked or input changed
    if recommend_clicked or st.session_state.get('ingredients_key', '') != '':
        if ingredients_input:
            try:
                ingredients = [ing.strip() for ing in ingredients_input.split(',')]
                recommendations = recommender.recommend_recipes(ingredients, top_n=num_recommendations)

                if not recommendations:
                    st.warning("No recipes found matching your ingredients.")
                    return

                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"Recipe {i}: {rec['recipe_name']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Similarity Score", f"{rec['similarity_score']:.2f}")
                        with col2:
                            overlap = recommender.ingredient_overlap(ingredients, rec['ingredients'])
                            st.metric("Ingredient Overlap", f"{overlap:.2f}%")

                        st.subheader("Ingredients")
                        ing_list = rec['ingredients'] if isinstance(rec['ingredients'], list) else eval(rec['ingredients_list'])
                        for ing in ing_list:
                            st.write(f"- {ing}")

                        st.subheader("Instructions")
                        steps_list = rec['steps'] if isinstance(rec['steps'], list) else eval(rec['steps'])
                        for j, step in enumerate(steps_list, 1):
                            st.write(f"{j}. {step}")

                        st.subheader("Nutritional Values (% of Daily Value)")
                        nutrition_values = rec['nutrition_values'] if isinstance(rec['nutrition_values'], list) else eval(rec['nutrition_values'])
                        nutrition_mapping = [
                            ("Calories", nutrition_values[0], ""),
                            ("Sugar", nutrition_values[2], "g"),
                            ("Sodium", nutrition_values[3], "mg"),
                            ("Protein", nutrition_values[4], "g"),
                            ("Saturated Fat", nutrition_values[5], "g"),
                            ("Carbohydrates", nutrition_values[6], "g")
                        ]
                        for name, value, unit in nutrition_mapping:
                            st.metric(name, f"{value:.2f} {unit}")

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter some ingredients!")

if __name__ == "__main__":
    main()