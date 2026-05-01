import streamlit as st
import pickle
import pandas as pd

st.write("Checking the engine... please wait a moment.")

movies_dict = pickle.load(open('movie_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl','rb'))
@st.cache_data
def load_data():
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    return movies, similarity

st.title('Movie Recommender System')

# Show a loading spinner so you know it's working
with st.spinner('Loading movie database...'):
    movies, similarity = load_data()

st.success('Ready to recommend!')

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

# UI Logic
st.title('Movie Recommender System')

selected_movie = st.selectbox(
    'Search for a movie:',
    movies['title'].values
)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    for i in recommendations:
        st.write(i)