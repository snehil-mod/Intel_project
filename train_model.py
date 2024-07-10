import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to training and validation directories
train_dir = r'E:\unnati intel Project\data\train'
val_dir = r'E:\unnati intel Project\data\validation'

# Create image data generators with basic augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Training and validation generators
train_gen = train_datagen.flow_from_directory(train_dir, target_size=(128, 128), batch_size=32, class_mode='binary')
val_gen = val_datagen.flow_from_directory(val_dir, target_size=(128, 128), batch_size=32, class_mode='binary')

# Load the model
model = load_model('simple_cnn_model.keras')

# Train the model
history = model.fit(train_gen, validation_data=val_gen, epochs=10)

# Save the trained model
model.save('trained_simple_cnn_model.keras')

# Save the training history
import pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
