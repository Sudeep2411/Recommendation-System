import streamlit as st
from emotion_detection import get_song_recommendations

def streamlit_interface():
    st.title("Emotion-Based Music Recommendation System")
    
    # Dropdown to select emotion
    emotion = st.selectbox("Select Emotion", ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])
    
    if st.button("Get Recommendations"):
        # Fetch recommendations for the selected emotion
        songs = get_song_recommendations(emotion)
        st.write(f"Recommended Songs for '{emotion}' Mood:")
        
        # Display song recommendations
        for song in songs:
            st.write(f"Genre: {song['Genre']}, Popularity: {song['Popularity']}")
            
if __name__ == "__main__":
    streamlit_interface()
