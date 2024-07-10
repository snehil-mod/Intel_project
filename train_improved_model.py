import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to training and validation directories
train_dir = r'E:\unnati intel Project\data\train'
val_dir = r'E:\unnati intel Project\data\validation'

# Create image data generators with enhanced augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Training and validation generators
train_gen = train_datagen.flow_from_directory(train_dir, target_size=(128, 128), batch_size=32, class_mode='binary')
val_gen = val_datagen.flow_from_directory(val_dir, target_size=(128, 128), batch_size=32, class_mode='binary')

# Load the improved model
model = load_model('improved_cnn_model.keras')

# Callbacks for early stopping, learning rate reduction, and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
model_checkpoint = ModelCheckpoint('best_improved_cnn_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# Train the model with callbacks
history = model.fit(train_gen, validation_data=val_gen, epochs=30, callbacks=[early_stopping, reduce_lr, model_checkpoint])

# Save the trained model
model.save('trained_improved_cnn_model.keras')

# Save the training history
import pickle
with open('improved_training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
