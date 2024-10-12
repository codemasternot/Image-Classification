#!/usr/bin/env python
# coding: utf-8

# In[87]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 
from tensorflow.keras.regularizers import l2
# Needed packages
train_dir = 'C:\\Users\\Stephen\\train'
test_dir = 'C:\\Users\\Stephen\\test'   # Train and test data paths


# In[88]:


train_datagen = ImageDataGenerator(
    rescale=1.0/255.0, # Play around with the parameters based on data complexity
    rotation_range=1,  # Reduce rotation depending on data size
    width_shift_range=0.005,  # Reduce horizontal shifts to improve generalization
    height_shift_range=0.005,  
    shear_range=0.005,  # Reduce shear transformations so more robust
    zoom_range=0.005,  # Reduce zoom range to help with scaling variations
    horizontal_flip=True,  # horizontal flip to learn symmetry
    fill_mode='nearest' 
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Only rescale for test/validation data


# In[89]:


# Load training data with image size increased to 224x224
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,  
    class_mode='categorical'
)

# Load test data with image size 214x214
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)


# In[90]:


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base_model layers to retain ImageNet weights
for layer in base_model.layers[-2:]:  # Unfreeze the last 2 layers to avoid overfitting based on data size
    layer.trainable = True


# In[91]:


# Add custom top layers for fine tuning
x = base_model.output
x = GlobalAveragePooling2D()(x)  
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.8)(x)  # Dropout to prevent overfitting
predictions = Dense(10, activation='softmax')(x)  

model = Model(inputs=base_model.input, outputs=predictions)


# In[92]:


model.compile(
    optimizer=Adam(learning_rate=0.00001),  # Small learning rate to prevent overfitting
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks/Early stopping and learning rate reduction on plateau
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6)


# In[106]:


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=25,  # Can increase or decrease depending on accuracy 
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    callbacks=[early_stop, reduce_lr]  
)


# In[103]:


# Save the trained model
model.save('C:\\Users\\Stephen\\Image_resnet50.h5')


# In[107]:


validation_loss, validation_acc = model.evaluate(test_generator)
print(f'Validation Accuracy: {validation_acc}') # Display val_accuracy


# In[ ]:




