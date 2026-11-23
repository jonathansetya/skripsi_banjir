from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import SGD

def build_cnn():
    model = Sequential()

    # Convolution Layer (3, 5, 1)
    model.add(Conv2D(16, (2,2), activation='relu', input_shape=(3, 5, 1)))
    model.add(MaxPooling2D(pool_size=(1,1)))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5)) # Naikkan dropout ke 50% agar tidak langsung menghafal

    # Output (3 Kelas)
    model.add(Dense(3, activation='softmax'))

    # Menggunakan SGD lambat agar grafik melengkung halus dari bawah ke atas
    model.compile(
        optimizer=SGD(learning_rate=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
