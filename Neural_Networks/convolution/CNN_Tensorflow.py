import tensorflow as tf
from tensorflow.keras import layers, models

# Definirea modelului CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'), # Test accuracy: 0.9894000291824341
    # layers.MaxPooling2D((2, 2)),
    # layers.Conv2D(64, (3, 3), activation='relu'),# Test accuracy: 0.9905999898910522 -> better compared to the above, here were 3 layers, the above had 64
    # layers.MaxPooling2D((2, 2)),
    # layers.Conv2D(128, (3, 3), activation='relu', padding='same'),    # Test accuracy: 0.9847999811172485 -> overfitting of the model, 4 layers with the above uncommented and lines 9-10 commented
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilarea modelului si optimizarea lui prin optimizatorul Adam
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Încărcarea setului de date MNIST
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocesarea datelor
train_images = train_images.reshape((60000, 28, 28, 1)) / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)) / 255.0

# Antrenarea modelului
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Evaluarea performanței modelului pe setul de date de testare
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
