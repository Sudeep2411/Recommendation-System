import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to the dataset
train_dir = "my_project/fer2013_dataset/train"
test_dir = "my_project/fer2013_dataset/test"

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(48, 48),
                                                    color_mode="grayscale", batch_size=64, 
                                                    class_mode="categorical")
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(48, 48),
                                                  color_mode="grayscale", batch_size=64,
                                                  class_mode="categorical")

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(7, activation="softmax")  # 7 emotion classes
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Training
model.fit(train_generator, validation_data=test_generator, epochs=10)

# Save the trained model
model.save("emotion_model.h5")
print("Model saved as emotion_model.h5")
