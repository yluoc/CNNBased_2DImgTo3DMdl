import tensorflow as tf

"""
Building a CNN model based on tensorflow,
it has 3 convolutional layers, 1 flatten layer, 2 connected layers,
you can adjust model as needed. 
"""

class cnnModel:
    def __init__(self, img_height, img_width, channels, output_vertices):
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.output_vertices = output_vertices
        self.model = self.build_model()

    def build_model(self):
        inputs = tf.keras.Input(shape=(self.img_height, self.img_width, self.channels), name='input_img')

        # convolutional layer 1
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
        pool1 = tf.keras.layers.MaxPooling2D((2, 2), padding='same', name='pool1')(conv1)
        drop1 = tf.keras.layers.Dropout(0.2, name='drop1')(pool1)

        # convolutional layer 2
        conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(drop1)
        pool2 = tf.keras.layers.MaxPooling2D((2, 2), padding='same', name='pool2')(conv2)
        drop2 = tf.keras.layers.Dropout(0.3, name='drop2')(pool2)

        # convolutional layer 3
        conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(drop2)
        pool3 = tf.keras.layers.MaxPooling2D((2, 2), padding='same', name='pool3')(conv3)
        drop3 = tf.keras.layers.Dropout(0.4, name='drop3')(pool3)

        # flatten layer
        flatten = tf.keras.layers.Flatten(name='flatten')(drop3)

        # fully connected layer 1
        dense1 = tf.keras.layers.Dense(1024, activation='relu', name='dense1')(flatten)
        drop4 = tf.keras.layers.Dropout(0.6, name='drop5')(dense1)

        # output layer (fully connected layer 1)
        dense2 = tf.keras.layers.Dense(self.output_vertices * 3, activation='linear', name='dense2')(drop4)
        outputs = tf.keras.layers.Reshape((self.output_vertices, 3), name='output_reshape')(dense2)

        # build model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')

        return model