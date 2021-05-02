# numpy is the data manipulation library that is needed.
import numpy as np
# tensorflow contains a lot of helpful/useful methods and data for machine
# learning
import tensorflow as tf
# os stands for operating system and it is used to run terminal commands like
# clearing the screen and other things of that sort.
import os

# stating the actual numbers that the neural network will be guessing
DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# loading in the neural network that was creating in another file.
with open('rps_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    json_file.close()
model = tf.keras.models.model_from_json(loaded_model_json)
model.load_weights("rps_model_weights.h5")


# creating the method that predicts the digit based on an inputed image.
def test_image(img):
    # reshaping the image to verify the image is compatible with the neural net
    img = img.reshape(-1, 28, 28)
    # making the image smaller so it will not be as hard on the neural network
    # grayscale image pixels have values from 0 to 255 (black to white)
    # dividing the image by 255 basically converts each pixel value from 0 to 
    # 255 too 0 to 1
    img = img/255.0
    # getting prediction percentages for each potential digit
    predictions = model.predict(img)
    # getting the position of the maximum value and accessing the digit in the
    # list create earilier
    prediction = DIGITS[np.argmax(predictions[0])]
    # clearing the screen/last value show
    os.system("cls")
    # displaying the neural nets prediction
    print(str(prediction))
