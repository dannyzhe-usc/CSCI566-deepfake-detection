import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
import numpy as np
import os

real_images_dir = 'Celeb-DF-v2/still_frames_input_cropped/real_input_cropped'
fake_images_dir = 'Celeb-DF-v2/still_frames_input_cropped/fake_input_cropped'

# hyperparams tune later 
image_size = (224, 224)  # resnet50 expects 224x224 images
batch_size = 32
epochs = 10

# image augmentations
# experiement with different number of augmentations later 
# experiment with using augmentations to address label imbalance (500 real vs. 60000 fake images)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    'Celeb-DF-v2/still_frames_input_cropped', 
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',  
    shuffle=True
)

# use class weights parameter to handle class imbalance
labels = np.array(train_generator.classes)
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# using resnet50 pretrained on imagenet for now
# research/experiment with different datasets for base model, consider base datasets of just people/faces
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# resnet50
model = models.Sequential([
    base_model,  
    layers.GlobalAveragePooling2D(), 
    layers.Dense(512, activation='relu'),  
    layers.Dropout(0.5),  
    layers.Dense(1, activation='sigmoid')  
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    class_weight=class_weight_dict  
)

model.save('resnet50_deepfake_classifier_with_class_weights.h5')
