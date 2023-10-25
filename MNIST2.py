import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
np.random.seed(42)
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam, SGD
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
print('Total no of Images: ',X_train.shape[0])
print('Size of Image:', X_train.shape[1:])
print('Total no of labels:', y_train.shape)
plt.imshow(X_train[0], cmap = plt.get_cmap('gray'))
print('Label:', y_train[0])
X_train = X_train.reshape((X_train.shape[0],-1))
X_test = X_test.reshape((X_test.shape[0], -1))

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print(X_train.shape, X_test.shape)

X_train = X_train/255
X_test = X_test/255


X_train.shape



y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)


num_classes = y_test.shape[1]
num_pixels = 784




def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(256, input_dim=num_pixels, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model


model = baseline_model()
model.summary()


opt = SGD(lr = 0.001)
model.compile(loss='categorical_crossentropy', optimizer= opt, metrics=['accuracy'])



model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

scores = model.evaluate(X_test, y_test, verbose=1)
print("Error: %.2f%%" % (100-scores[1]*100))

img_width, img_height = 28, 28

ii = cv2.imread("D:/img_77.jpg")
gray_image = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
# print(gray_image)
plt.imshow(gray_image,cmap='Greys')
plt.show()

x = np.expand_dims(gray_image, axis=0)
x = x.reshape((1, -1))



pred=model.predict(x)
classes=np.argmax(pred,axis=1)



print('Predicted value is ',pred[0])
