# Context-Aware Music Recommendation System

## Overview
The Context-Aware Music Recommendation System is designed to recommend music based on real-time emotion recognition from facial expressions. It integrates advanced facial recognition and music recommendation technologies to provide a seamless and personalized user experience.

## Features
1. **Mood Classification**: Uses a Convolutional Neural Network (CNN) trained on the FER-2013 dataset to classify emotions.
2. **Real-Time Mood Detection**: Leverages webcam input to detect user mood in real-time.
3. **Music Recommendation**: Matches detected moods with appropriate songs from the Million Song dataset.
4. **User Interface**: Built with Streamlit for an intuitive and accessible interface.

## Technologies Used
- **Python Libraries**: TensorFlow, Keras, NumPy, OpenCV, Pandas
- **Datasets**:
  - FER-2013 for mood classification
  - Million Song Dataset for music recommendation
- **Frameworks**: Streamlit for UI development

## Project Structure
```
my_project/
├── fer2013_dataset/
│   ├── train/
│   └── test/
├── spotify_data.csv
├── mood_detection.py
├── music_recommendation.py
├── streamlit_app.py
├── fer2013.h5
├── requirements.txt
├── README.md
└── venv/
```

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd my_project
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage
1. Launch the application by running the Streamlit app.
2. Use your webcam to allow the system to detect your mood.
3. Receive real-time music recommendations based on your detected mood.

## Contributing
Contributions are welcome! Feel free to submit issues and pull requests for enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- **FER-2013 Dataset**: For enabling accurate mood classification.
- **Million Song Dataset**: For providing comprehensive music data.
- **Streamlit**: For creating a user-friendly interface.
