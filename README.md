# EX07-Convolutional-Autoencoder-for-Image-Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

 The MNIST dataset, which is a simple computer vision dataset. It consists of images of handwritten digits in the form of a greyscale. It also includes 
 labels for each image, telling us which digit it is . It means we have a labelled data in our hands to work with. 
 Each image in the MNIST dataset is 28 pixels by 28 pixels.

## DESIGN STEPS

### STEP 1:

 Import the requried libraries for further operation, then load the dataset which is a simple computer vision dataset called MNIST

### STEP 2:

 Then reshape the loaded images, then add some nosiy to the image it seams the image are converted to pixel, Encode the particular image using various layers
 on the model.

### STEP 3:

 One of the possible reasons for this blurriness in the reconstructed images is to use fewer epochs while training our model. Therefore, now it’s your task to 
 increase  the value of the number of epochs and then again observed these images and also compare with these.

## PROGRAM

```python3
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
```

```python3
(x_train, _), (x_test, _) = mnist.load_data()
x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))
noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape) 
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape) 
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
model=Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(16, (5,5), activation='relu'),
    layers.MaxPool2D((2,2), padding='same'),
    layers.Conv2D(8, (3,3), activation='relu'),
    layers.MaxPool2D((2,2), padding='same'), 
    layers.Conv2D(8, (3,3), activation ='relu', padding='same'),
    layers.UpSampling2D((2,2)),
    layers.Conv2D(16, (5,5), activation='relu', padding='same'),
    layers.UpSampling2D((3,3)),
    layers.Conv2D(1, (3,3), activation="sigmoid")
])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(x_train_noisy, x_train_scaled,epochs=2,batch_size=128,shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))
```

```python3
decoded_imgs = model.predict(x_test_noisy)
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

## OUTPUT

### Original vs Noisy Vs Reconstructed Image

![Screenshot (52)](https://user-images.githubusercontent.com/89703145/202248493-8870dcf9-5a27-46fd-8f2c-affd0dccbd54.png)

## RESULT
 Thus, we have built an autoencoder model, which can successfully clean very noisy images, which it has never seen before
