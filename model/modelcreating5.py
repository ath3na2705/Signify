import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append(r"c:\users\a0105\appdata\local\programs\python\python313\lib\site-packages")

# Check TensorFlow version
print(tf.__version__)

# Data Augmentation (more balanced to avoid overfitting)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),  # Reduced rotation to avoid unrealistic scenarios
    tf.keras.layers.RandomZoom(0.1)  # Reduced zoom to avoid drastic changes
])

# Load the dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\a0105\Downloads\archive\asl_alphabet_train\asl_alphabet_train",
    image_size=(180, 180),  # Resize images to a uniform size
    batch_size=32,
    shuffle=True,  # Ensure data is shuffled before splitting
    seed=123  # Ensures reproducibility when shuffling
).map(lambda x, y: (x / 255.0, y))  # Normalize images

# Determine the size of the dataset
dataset_size = tf.data.experimental.cardinality(dataset).numpy()

# Split the dataset into training and validation sets (80% training, 20% validation)
train_size = int(0.8 * dataset_size)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# Apply data augmentation only to the training dataset
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

# Prefetch the datasets to improve performance
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Check class distribution in the training dataset
class_counts = np.zeros(29)
for _, labels in train_dataset:
    class_counts += np.bincount(labels.numpy(), minlength=29)

print("Class distribution:", class_counts)

# Adjust class weights for imbalanced classes
class_weights = {i: max(1.0 / class_counts[i], 1.0) for i in range(29)}  # Clip weights to avoid extreme values

# Define the model architecture with batch normalization and regularization
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # L2 regularization
    tf.keras.layers.Dropout(0.5),  # Helps prevent overfitting
    tf.keras.layers.Dense(29, activation='softmax')  # 29 classes
])

# Print model summary
model.summary()

# Compile the model with a lower learning rate for the Adam optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Add a learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return float(lr * tf.math.exp(-0.1))

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Train the model with class weights and learning rate scheduler
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    class_weight=class_weights,
    callbacks=[lr_scheduler]
)

# Save the model
model.save("signify_ASL_image_classification_model_ver4.keras")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Load the test dataset
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\a0105\Downloads\archive\asl_alphabet_test\asl_alphabet_test",
    image_size=(180, 180),
    batch_size=32
).map(lambda x, y: (x / 255.0, y))  # Normalize images

# Prefetch the test dataset
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy}")


