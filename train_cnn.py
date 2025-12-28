import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# --- Data Augmentation ---
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    'data/dataset',
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    'data/dataset',
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# --- Pretrained MobileNetV2 ---
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

# Freeze base layers
base_model.trainable = False

# --- Add Custom Layers ---
inputs = Input(shape=(224,224,3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(4, activation='softmax')(x)  # 4 classes
model = Model(inputs, outputs)

# --- Compile Model ---
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- Callbacks ---
os.makedirs('artifacts', exist_ok=True)
checkpoint = ModelCheckpoint('artifacts/cnn_rice_leaf_transfer.keras', 
                             monitor='val_accuracy', save_best_only=True)
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# --- Train ---
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[checkpoint, early_stop]
)

print("Training completed. Model saved at artifacts/cnn_rice_leaf_transfer.keras")
