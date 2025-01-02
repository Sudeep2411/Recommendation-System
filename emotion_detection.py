import cv2
import numpy as np
from keras.models import load_model
import pandas as pd

# Load the pre-trained emotion detection model
emotion_model = load_model("emotion_model.h5")
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

# Load the Spotify dataset
spotify_data = pd.read_csv("spotify_data.csv")

# Map emotions to genres
emotion_to_genre = {
    "Angry": "rock",
    "Happy": "pop",
    "Sad": "acoustic",
    "Neutral": "classical",
    "Surprise": "dance",
    "Fear": "ambient",
    "Disgust": "jazz"
}

def detect_emotion(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    emotion = "Neutral"  # Default emotion in case no face is detected
    
    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        
        prediction = emotion_model.predict(roi_gray)
        emotion = emotion_dict[np.argmax(prediction)]  # Get the predicted emotion
    
    return emotion

def get_song_recommendations(emotion):
    genre = emotion_to_genre.get(emotion, "pop")
    recommendations = spotify_data[spotify_data["Genre"] == genre]
    top_songs = recommendations.sort_values("Popularity", ascending=False).head(5)
    return top_songs[["Genre", "Popularity"]].to_dict(orient="records")

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotion = detect_emotion(frame)
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Get song recommendations for the detected emotion
        songs = get_song_recommendations(emotion)
        print(f"Detected Emotion: {emotion}")
        print("Recommended Songs:")
        for song in songs:
            print(f"Genre: {song['Genre']}, Popularity: {song['Popularity']}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
