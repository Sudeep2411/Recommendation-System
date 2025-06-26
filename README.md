# 🎵 Context-Aware Music Recommendation System

## Overview

The **Context-Aware Music Recommendation System** leverages real-time **facial emotion recognition** to suggest music that aligns with the user’s mood. It combines **Computer Vision**, **Deep Learning**, and **Recommendation Systems** to create a seamless, interactive, and personalized music experience.

---

## 🚀 Features

- 🎭 **Emotion Detection**: Classifies facial expressions into 7 emotions using a CNN trained on the FER-2013 dataset.
- 🎧 **Music Recommendation**: Matches detected emotions with the most suitable music genres and songs using the Million Song Dataset.
- 📷 **Real-Time Mood Detection**: Uses a webcam feed to detect user emotions on the fly.
- 🌐 **Streamlit Interface**: A simple and accessible web UI for selecting emotions and viewing recommendations.

---

## 🧠 Tech Stack

| Component            | Technology                                  |
|----------------------|---------------------------------------------|
| Language             | Python                                      |
| Deep Learning        | TensorFlow, Keras                           |
| Image Processing     | OpenCV                                      |
| Data Handling        | Pandas, NumPy                               |
| UI Framework         | Streamlit                                   |
| Datasets             | FER-2013 (Facial Emotions), Spotify (Kaggle)|
| Model Output         | `emotion_model.h5` (saved trained model)    |

---

## 🗂️ Project Structure

```
my_project/
├── fer2013_dataset/            # FER-2013 training/test data
│   ├── train/
│   └── test/
├── spotify_data.csv            # Spotify song dataset
├── train_emotion.py            # CNN training script
├── music_recommendation.py     # Real-time detection and recommendation
├── streamlit_app.py            # Streamlit-based UI
├── fer2013.h5                  # Saved trained CNN model
├── requirements.txt            # Required Python packages
├── README.md                   # Project documentation
└── venv/                       # Python virtual environment
```

---

## ⚙️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd my_project
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train Emotion Model (Optional)**
   ```bash
   python train_emotion.py
   ```

5. **Run Real-Time Recommendation App**
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 💡 How It Works

### Emotion Detection
- Uses OpenCV to capture webcam frames.
- Detects faces and preprocesses them for CNN.
- The CNN model (`emotion_model.h5`) predicts the facial emotion.

### Music Recommendation
- Maps predicted emotion to a music genre.
- Filters and sorts the top 5 songs by popularity from `spotify_data.csv`.

### Web App
- Users can either use webcam-based emotion detection or manually select an emotion.
- Streamlit displays genre-based song recommendations.

---


## 📈 Sample Output

```text
Detected Emotion: Happy
Recommended Songs:
Genre: pop, Popularity: 98
Genre: pop, Popularity: 97
...
```

---

## 🧪 Requirements

Sample `requirements.txt`:

```txt
tensorflow
keras
opencv-python
numpy
pandas
streamlit
```

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome!  
Please open an issue or submit a pull request for any suggestions or improvements.

---

## 📄 License

This project is licensed under the **MIT License**. See the [`LICENSE`](./LICENSE) file for details.

---

## 🙏 Acknowledgments

- [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013): For facial emotion classification.
- [Million Song Dataset / Spotify Dataset](https://www.kaggle.com/datasets/): For music recommendation.
- [Streamlit](https://streamlit.io): For the interactive web UI.
