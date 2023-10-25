import itertools
import os

from sklearn.model_selection import train_test_split
import keras
#import tensorflow as tf
import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, Flatten, MaxPooling1D
from keras import layers




np.random.seed(3)

# number of wine classes
classifications = 2

data_dim = 32
timesteps = 1

def plot_confusion_matrix(name,cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(2, figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    os.chdir('/..')
    plt.savefig(f'D:/JHML-Python-Projects/{name}-Confusion-Matrix.png')

def evaluate_model(model,x_test,y_test,name):
    print(f'Evaluating {name} model..........')
    score = model.evaluate(x_test, y_test, batch_size=10, verbose=2)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def predict_model(model,x_test,y_test,name):
    print(f'Predicting {name} model..........')
    predictions=model.predict(x_test,batch_size=10)
    pred=np.argmax(predictions,axis=1)
    print(f'{name} Prediction Score...........')
   # print(classification_report(y_test,pred))
    cm = confusion_matrix(y_true=y_test, y_pred=pred)
    cm_plot_labels=[str(i) for i in range(10)]
    plot_confusion_matrix(name,cm=cm, classes=cm_plot_labels, title=f'{name} Confusion Matrix')


# load dataset
dataset = np.loadtxt('D:/Maldroid_256_class4.csv', delimiter=",")
print("Parkinson Dataset Shape:",dataset.shape)
print("First five samples:\n",dataset[1:5,])



# split dataset into sets for testing and training
X = dataset[1:,0:32]
Y = dataset[1:,32]
print("X shape:",X.shape)
print("Y shape:",Y.shape)
print("Targets:",Y)

print()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=5)
print("x_train:",len(x_train))
print("x_test:",len(x_train))
print("y_train:",len(x_train))
print("y_test:",len(x_train))





#print(x_train.shape,y_train.shape)
#print(x_test.shape,y_test.shape)
#x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)

X_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
X_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
print("X_train after reshape:",X_train.shape)
print("X_test after reshape:",X_test.shape)

# convert output values to one-hot
y_train = keras.utils.np_utils.to_categorical(y_train)
y_test = keras.utils.np_utils.to_categorical(y_test)
print("y_train:",y_train[1:5])
print("y_test:",y_test[1:5])


# creating model
model = Sequential()
model.add(LSTM(60, input_shape=(timesteps, data_dim), activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(40, activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(40, activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation= 'relu'))

model.add(Dense(classifications, activation='sigmoid'))
model.summary()
# compile and fit model
#model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
#model.compile(loss="mean_squared_error", optimizer="adam")
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=15, epochs=30, validation_data=(X_test, y_test))
#history = model.fit(X_train, y_train, batch_size=15, epochs=20, validation_split= 0.2)
print(history.history.keys())
pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)
y_test2 = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test2, pred)
np.set_printoptions(precision = 2)
print('confusion matrix without normalization')
print(cm)

fig, ax = plt.subplots()
im = ax.imshow(cm, cmap=plt.get_cmap('hot'), interpolation='nearest',
               vmin=0, vmax=1)
fig.colorbar(im)
plt.show()
#skplt.metrics.plot_confusion_matrix(cm)

#fig, ax = plot_confusion_matrix(conf_mat=cm,
#                           colorbar=True,
#                           show_absolute=False,
#                         show_normed=True)

#plt.imshow(cm)
plt.figure()
cm_normalized = cm.astype('float') / cm.sum(axis=1) [:, np.newaxis]
print('confusion matrix with normalization')
print(cm_normalized)
print(y_test2)
print(pred)


plt.figure()


print('results')
print('Accuracy Score :'), accuracy_score(y_test2, pred)
print('Report : ')
print(classification_report(y_test2, pred))
#cr = classification_report(y_test2, pred)
#plot_classification_report(cr)
print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();



evaluate_model(model,X_test,y_test,'DNN')
predict_model(model,X_test,y_test,'DNN')

