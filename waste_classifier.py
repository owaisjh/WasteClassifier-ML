from keras.models import load_model
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle

def start(n):
    if(n==1):
        MODEL_FILENAME = "trash_model.hdf5"
        MODEL_LABELS_FILENAME = "model_labels.dat"
        CAPTCHA_IMAGE_FOLDER = "testtrash"


        # Load up the model labels (so we can translate model predictions to actual letters)
        with open(MODEL_LABELS_FILENAME, "rb") as f:
            lb = pickle.load(f)

        # Load the trained neural network
        model = load_model(MODEL_FILENAME)

        # lets input the captcha image to test our network
        captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
        captcha_image_files = np.random.choice(captcha_image_files, size=(1,), replace=False)
        #captcha_image_files = list(captcha_image_files)
        #captcha_image_files.append('sample_captcha2.png')
        #captcha_image_files = np.array(captcha_image_files)

        print(captcha_image_files)
        # loop over the image paths
        for image_file in captcha_image_files:
            # Load the image and convert it to grayscale
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Add some extra padding around the image
           # image = cv2.copyMakeBorder(image, 100,100, 100, 100, cv2.BORDER_REPLICATE)

            # threshold the image (convert it to pure black and white)
            thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            # find the contours (continuous blobs of pixels) the image
            contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

            letter_image_regions = []
            # print(letter_image_regions)

            # Get the rectangle that contains the contour
            (x, y, w, h) = cv2.boundingRect(contours[0])
            letter_image_regions.append((x, y, w, h))

            # Create an output image and a list to hold our predicted letters
            output = cv2.merge([image] * 3)
            predictions = []

            # loop over the letters
            x, y, w, h = letter_image_regions[0]

            # Extract the letter from the original image with a 2-pixel margin around the edge
            letter_image = image[y - 2500:y + h + 2500, x - 2500:x + w + 2500]

            # Re-size the letter image to 20x20 pixels to match training data
            letter_image = cv2.resize(letter_image, (20, 20))

            # Turn the single image into a 4d list of images to make Keras happy
            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)

            # Ask the neural network to make a prediction
            prediction = model.predict(letter_image)

            # Convert the one-hot-encoded prediction back to a normal letter
            letter = lb.inverse_transform(prediction)[0]
            predictions.append(letter)

            # draw the prediction on the output image
            cv2.rectangle(output, (x-2, y-2), (x + w, y ), (0, 255, 0), 1)
            cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Print the captcha's text
        captcha_text = "".join(predictions)
        print("CAPTCHA text is: {}".format(captcha_text))

            # Show the annotated image
        # cv2.imshow("Output", output)
        return output
        cv2.waitKey()


