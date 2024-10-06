#!/usr/bin/env python
# coding: utf-8

# In[87]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # Correct import for mixed precision
from tensorflow.keras.regularizers import l2
# Paths to your dataset
train_dir = 'C:\\Users\\Stephen\\train'
test_dir = 'C:\\Users\\Stephen\\test'   # Example: 'data/test'


# In[88]:


train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=1,  # Further reduce rotation
    width_shift_range=0.005,  # Reduce horizontal shifts
    height_shift_range=0.005,  # Reduce vertical shifts
    shear_range=0.005,  # Reduce shear transformations
    zoom_range=0.005,  # Reduce zoom range
    horizontal_flip=True,  # Keep horizontal flip
    fill_mode='nearest'  # Fill empty pixels after transformations
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Only rescale for test/validation data


# In[89]:


# Load training data from directory with image size increased to 224x224
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,  # Adjust batch size based on your memory constraints
    class_mode='categorical'
)

# Load test data with image size 512x512
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)


# In[90]:


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base_model layers to retain pre-trained ImageNet weights
for layer in base_model.layers[-2:]:  # Unfreeze the last 50 layers
    layer.trainable = True


# In[91]:


# Add custom top layers for fine-tuning
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Efficient pooling operation after ResNet50
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.8)(x)  # Dropout to prevent overfitting
predictions = Dense(10, activation='softmax')(x)  # Output layer for 10 categories

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)


# In[92]:


model.compile(
    optimizer=Adam(learning_rate=0.00001),  # Small learning rate for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks: Early stopping and learning rate reduction on plateau
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6)


# In[106]:


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=1,  # You can increase or decrease based on validation results
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    callbacks=[early_stop, reduce_lr]  # Early stopping and learning rate scheduler
)


# In[103]:


# Save the trained model
model.save('C:\\Users\\Stephen\\Image_resnet50.h5')


# In[107]:


validation_loss, validation_acc = model.evaluate(test_generator)
print(f'Validation Accuracy: {validation_acc}')


# In[ ]:




