# cv2 is used for a lot of the image processing that needs to be done
import cv2
# PIL is used for a smaller ammount of necesary image processing
from PIL import Image
# numpy is a library used to manipulate numbers which can make the images 
# more friendly to the neural net
import numpy as np
# the neural_net is another file that contains the actuall neural network 
# itself.
import neural_net

# Getting a connection to the camera.
vid_cap = cv2.VideoCapture(1)

# creating a method to process the image
def process_img(original_img):
    # converting image to grayscale
    processed_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    # simplifing the image into one line (that is the digit)
    processed_img = cv2.Canny(processed_img, threshold1=175, threshold2=285)
    # returning the new processed image
    return processed_img

# running the program forever. (While True: basically just means while 
# True = True do everything indented below it)
while True:
    # Get camera feed from the connection created earlier
    has_feed, frame = vid_cap.read()

    # Crop the frame into 500x500px
    image = Image.fromarray(frame).crop((575, 280, 775, 480))

    # Turning the image into a numpy array which is basically just a list of 
    # all the image data it is basically a list or rows, which are a list of 
    # pixel values. This makes it easier for the neural net to read
    image_array = np.array(image)

    # resizing this image to 28x28px because the smaller the image, the 
    # easier it can be processed by the neural net
    image_array = cv2.resize(image_array, (28, 28))

    # this puts the image throug the processing method that was created earlier
    final_image_array = process_img(image_array)

    # this is the method that puts the image through the neural network
    neural_net.test_image(final_image_array)

    # this ouputs a video of the camera feed with a nice box where the image
    # is cropped
    show_frame = cv2.rectangle(frame, (575, 280), (775, 480), (50, 245, 0), 2)
    cv2.imshow("Window", show_frame)

    # this line basically says that if the q key is pressed the program will
    # shut down
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# this closes the connection to the camera
vid_cap.release()
# and destorys all of the video windows.
cv2.destroyAllWindows()
