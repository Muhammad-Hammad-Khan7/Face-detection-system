#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.layers import Input, Dense, Flatten,Lambda
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# In[ ]:


import tensorflow as tf
import os, shutil, zipfile

# 1. Download dataset
url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
path_to_zip = tf.keras.utils.get_file("cats_and_dogs_filtered.zip", origin=url)

# 2. Extract manually
with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
    zip_ref.extractall(os.path.dirname(path_to_zip))

# 3. Now base_dir will exist
base_dir = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
print("Base directory:", base_dir)
print("Contents:", os.listdir(base_dir))

# 4. Paths for train & validation
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

# 5. Create folders your code expects
os.makedirs("Datasets/Train", exist_ok=True)
os.makedirs("Datasets/Test", exist_ok=True)

# 6. Copy train data -> Datasets/Train
for cls in os.listdir(train_dir):
    src = os.path.join(train_dir, cls)
    dst = os.path.join("Datasets/Train", cls)
    if not os.path.exists(dst):
        shutil.copytree(src, dst)

# 7. Copy validation data -> Datasets/Test
for cls in os.listdir(val_dir):
    src = os.path.join(val_dir, cls)
    dst = os.path.join("Datasets/Test", cls)
    if not os.path.exists(dst):
        shutil.copytree(src, dst)

print("✅ Dataset ready!")
print("Train classes:", os.listdir("Datasets/Train"))
print("Test classes:", os.listdir("Datasets/Test"))



# In[ ]:


image_size = [224,224]

# In[ ]:


train_path = 'Datasets/Train'
valid_path = 'Datasets/Test'

# In[ ]:


# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=image_size + [3], weights='imagenet', include_top=False)



# In[ ]:


# don't train existing weights
for layer in vgg.layers:
  layer.trainable = False

# In[ ]:


# useful for getting number of classes
folders = glob('.train/*')

# In[ ]:


# our layers - you can add more if you want
x = Flatten()(vgg.output)
# x = Dense(2, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# In[ ]:


# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# In[ ]:


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Datasets/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Datasets/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')



# In[ ]:


'''r=model.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 2000)'''



# In[ ]:


r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# In[ ]:


# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# In[ ]:


# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[ ]:


import tensorflow as tf

from keras.models import load_model

model.save('facefeatures_new_model.h5')

