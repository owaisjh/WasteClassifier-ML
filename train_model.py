import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten

TRASH_IMAGES_FOLDER = "extracted_trash"
MODEL_FILENAME = "trash_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"

# Initialise the data and labels
data = []
labels = []

# Loop over the input images and manipulate it
for image_file in paths.list_images(TRASH_IMAGES_FOLDER):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (20, 20), cv2.INTER_AREA)
    image = np.expand_dims(image, axis=2)

        # Grab the type of trash and append it and image in a list
    label = image_file.split(os.path.sep)[1]
    #print(label[1])
    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# Split into training and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Convert the labels (trash classification) into one-hot encodings ofr keras
lb = LabelBinarizer().fit(list(Y_train) + list(Y_test))
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# save the mappings from labels to one_hot_encodings
# We will use this later when we use the model to decode the prediction
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

### Build the Neural network ###
model = Sequential()
model.add(Conv2D(20, (20, 20), padding="same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(50, (5,5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation="relu"))
model.add(Dense(5, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the neural network
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=64, epochs=100, verbose=1)

# Summary of the model
print(model.summary())
# Save the trained model to disk
model.save(MODEL_FILENAME)

