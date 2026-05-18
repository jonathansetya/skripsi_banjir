from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    Flatten,
    Dense,
    MaxPooling2D
)

def build_cnn():

    model = Sequential()

    # Convolution Layer
    model.add(
        Conv2D(
            8,
            (1,1),
            activation='relu',
            input_shape=(1,1,1)
        )
    )

    # Pooling Layer
    model.add(
        MaxPooling2D(pool_size=(1,1))
    )

    # Flatten
    model.add(Flatten())

    # Fully Connected
    model.add(Dense(16, activation='relu'))

    # Output
    model.add(Dense(3, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model