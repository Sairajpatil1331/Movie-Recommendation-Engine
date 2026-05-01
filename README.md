# Content-Based Movie Recommendation System 🎬

A machine learning-powered web application that recommends movies based on their metadata (genres, keywords, cast, and crew) using Natural Language Processing (NLP) and Cosine Similarity.

## 🚀 Project Overview
This project was built as part of my Data Science internship curriculum. It utilizes the **TMDB 5000 Movie Dataset** to suggest films that are mathematically "closest" to a user's selection.

## 🧠 How It Works
The engine follows a content-based filtering approach:
1. **Data Preprocessing**: Merging movies and credits datasets, handling JSON-formatted metadata, and extracting key features (Director, Top 3 Actors, Genres, and Keywords).
2. **Text Cleaning**: Removing spaces from names (e.g., "Christopher Nolan" to "ChristopherNolan") and applying **Porter Stemming** to normalize words.
3. **Vectorization**: Using `CountVectorizer` to convert text tags into a 5,000-dimensional vector space.
4. **Similarity Scoring**: Calculating the **Cosine Similarity** between movie vectors to determine proximity.



## 🛠️ Tech Stack
* **Language**: Python
* **Libraries**: Pandas, NumPy, Scikit-Learn, NLTK
* **Frontend**: Streamlit
* **Deployment**: Localhost (via Streamlit)

## 📊 The Mathematics
To find similar movies, we calculate the cosine of the angle between two vectors:

$$ \text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} $$



## 📋 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/Sairajpatil1331/Movie-Recommendation-Engine.git](https://github.com/Sairajpatil1331/Movie-Recommendation-Engine.git)
Install dependencies:

Bash

pip install -r requirements.txt
Run the Jupyter Notebook (model_building.ipynb) to generate the movie_dict.pkl and similarity.pkl files.

Launch the Streamlit app:

Bash

streamlit run app.py
📁 Project Structure
model_building.ipynb: Data cleaning, NLP, and model generation.

app.py: Streamlit web interface.

data/: Folder containing raw CSV files (not uploaded due to size).

*.pkl: Serialized model files (ignored by git due to size).


---

### Why this README works:
* **The "How it Works" Section**: Explains your logic clearly for technical interviews.
* **Math Formulas**: Using LaTeX for the Cosine Similarity formula makes your repo look academically rigorous.
* **Setup Instructions**: It clarifies that the user needs to run the notebook first to generate the `.pkl` files, which explains why those files aren't in the GitHub repo.

### One Final Step: `requirements.txt`
To make your README's "How to Run" section work, create one more file in VS Code named `requirements.txt` and paste these lines:
```text
pandas
numpy
scikit-learn
nltk
streamlit
