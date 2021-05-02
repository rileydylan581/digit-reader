import tensorflow as tf

digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

training_data = tf.keras.datasets.mnist  # Data set of 28x28 images of handwritten digits and their labels
(x_train, y_train), (x_test, y_test) = training_data.load_data()  # Unpacks images and labels

x_train = tf.keras.utils.normalize(x_train, axis=1)  # Scales data between 0 and 1

model = tf.keras.models.Sequential()  # a basic feed-forward model

def create_model():
    model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
    model.add(
        tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
    model.add(
        tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
    model.add(tf.keras.layers.Dense(10,
                                    activation=tf.nn.softmax))  # our output layer. 10 units for 10 classes. Softmax for
    # probability distribution


def train_model():
    create_model()
    model.compile(optimizer='adam',  # Good default optimizer to start with
                  loss='sparse_categorical_crossentropy',
                  # how will we calculate our "error." Neural network aims to minimize loss.
                  metrics=['accuracy'])  # what to track

    model.fit(x_train, y_train, epochs=3)  # train the model


def save_model():
    train_model()
    model_json = model.to_json()
    with open("rps_model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("rps_model_weights.h5")
    print("Saved model to disk")


save_model()


def load_model():
    json_file = open('rps_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights("rps_model_weights.h5")
