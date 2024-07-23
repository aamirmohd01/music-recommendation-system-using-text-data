from flask import Flask, render_template, request, jsonify,redirect,session
from flask_session import Session
import pickle
import numpy as np
from tensorflow.keras.utils import pad_sequences # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
 
 
@app.route("/")
def index():
    if not session.get("user_id"):
        return redirect("/login")
    return render_template('index.html')
 
 
@app.route("/login", methods=["POST","GET"])
def login():
    # print(session["user_id"])
    if session.get("user_id"):
        return redirect("/")
    if request.method == "POST":
        session["name"] = request.form.get("name")
        if not get_user_id_by_name(session["name"]):
            context = {"message":"User not found or incorrect password"}
            return render_template("login.html",**context)
        else:
            session["user_id"] = get_user_id_by_name(session["name"]) 
            session["user_id"] = session["user_id"][0][0]
            return redirect("/")
    return render_template("login.html")
 
 
@app.route("/logout")
def logout():
    session["name"] = None
    session["user_id"] = None
    return redirect("/login")


def get_user_id_by_name(name):
    try:
        db = connect_db()
        cursor = db.cursor()
        query = """
        SELECT user_id from users where username = %s;
        """
        cursor.execute(query,(name,))
        data = cursor.fetchall()
        cursor.close()
        db.close()
        return data
    except Exception as e:
        print(e)


def get_liked_songs(user_id, song_ids):
    # Example SQL query to check liked songs
    # You should replace this with your actual query
    db = connect_db()
    cursor = db.cursor()
    liked_songs_query = """
    SELECT song_id
    FROM user_likes
    WHERE user_id = %s
    AND song_id IN ({})
    """.format(','.join(map(str, song_ids)))  # Convert song_ids to string for SQL query
    cursor.execute(liked_songs_query,(user_id,))
    liked_song_ids = [row[0] for row in cursor.fetchall()]
    cursor.close()
    db.close()
    # Generate liked_songs list based on whether each song_id is in liked_song_ids
    liked_songs = [1 if song_id in liked_song_ids else 0 for song_id in song_ids]
    return liked_songs





def load_tokenizer():
    return pickle.load(open('tokenizer_new.pickle', 'rb'))

def load_modl():
    return load_model('lyrics_topic_model_new.h5')

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="music_recommendation"
    )




def fetch_all_songs():
    try:
        db = connect_db()
        cursor = db.cursor()
        query = """
        SELECT songs.song_id, songs.title, songs.artist, songs.genre, songs.emotion FROM songs
        """
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()
        db.close()
        return pd.DataFrame(data, columns=['song_id', 'song_title', 'artist', 'genre', 'emotion'])
    except Exception as e:
        print(e)
        return pd.DataFrame()

def fetch_user_preference(user_id):
    try:
        db = connect_db()
        cursor = db.cursor()
        query = """
        SELECT DISTINCT s.artist, s.genre FROM user_likes ul
        JOIN songs s ON ul.song_id = s.song_id
        WHERE ul.user_id = %s;
        """
        cursor.execute(query, (user_id,))
        data = cursor.fetchall()
        cursor.close()
        db.close()
        return pd.DataFrame(data, columns=['favorite_artists', 'preferred_genres'])
    except Exception as e:
        print(e)
        return pd.DataFrame()

def recommend_songs(user_profile, cosine_sim, filtered_df):
    favorite_artists = user_profile['favorite_artists'].tolist()
    preferred_genres = user_profile['preferred_genres'].tolist()

    results = []

    for idx, row in filtered_df.iterrows():
        artist_sim = 1 if row['artist'] in favorite_artists else 0
        genre_sim = 1 if row['genre'] in preferred_genres else 0
        similarity_score = (artist_sim + genre_sim) / 2

        results.append({
            'song_id': row['song_id'],
            'song_title': row['song_title'],
            'artist': row['artist'],
            'similarity_score': similarity_score,
        })

    results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)

    return results



@app.route('/', methods=['POST'])
def predict():
    para = request.form.get('para')
    tokenizer = load_tokenizer()
    sequences = tokenizer.texts_to_sequences([para])
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=35)
    model = load_modl()
    prediction = model.predict(padded_sequences)
    predicted_topic = np.argmax(prediction)
    emotion_map = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Anger', 4: 'Fear', 5: 'Surprise'}
    predicted_emotion = emotion_map[predicted_topic]

    df = fetch_all_songs()
    filtered_df = df[df['emotion'].str.lower() == predicted_emotion.lower()]

    tfidf = TfidfVectorizer(stop_words='english')
    features = tfidf.fit_transform(filtered_df['genre'] + ' ' + filtered_df['artist'])
    cosine_sim = cosine_similarity(features, features)

    user_id = session["user_id"]
    user_profile = fetch_user_preference(user_id)
    results = recommend_songs(user_profile, cosine_sim, filtered_df)

    song_ids = [song["song_id"] for song in results[:5]]
    # print(song_ids)
    liked_songs = get_liked_songs(user_id, song_ids)
    # return "Your emotion is: " + predicted_emotion
    # print(liked_songs)
    ret_response = {
        'predicted_emotion': predicted_emotion,
        'results': results[:5],  # Sending top 5 recommendations
        'liked_songs':liked_songs,
    }

    return ret_response

@app.route('/toggle_like',methods=['POST'])
def toggle_like():
    user_id = session["user_id"]

    try:
        song_id = int(request.form.get('song_id'))
        # print(song_id)
        action = request.form.get('action')  # 'like' or 'unlike'
        if action=="like":
            # print("like")
            try:
                db = connect_db()
                cursor = db.cursor()
                query = """
                INSERT INTO user_likes (user_id,song_id) values (%s,%s);
                """
                cursor.execute(query, (user_id,song_id))
                cursor.close()
                db.commit()
                db.close()
                return jsonify({'success': True}), 200
                
            except Exception as e:
                print(e)
                return jsonify({'success': False, 'error': str(e)}), 500
        elif action=="unlike":
            # print("unlike")
            try:
                db = connect_db()
                cursor = db.cursor()
                query = """
                DELETE from user_likes where user_id = %s and song_id = %s;
                """
                cursor.execute(query, (user_id,song_id))
                cursor.close()
                db.commit()
                db.close()
                return jsonify({'success': True}), 200

            except Exception as e:
                print(e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
    except Exception as e:
        print(e)
        return jsonify({'success': False, 'error': str(e)}), 500
    
    return jsonify({'success': True}), 200


if __name__ == '__main__':
    app.run(debug=True)
