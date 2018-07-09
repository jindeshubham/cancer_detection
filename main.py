import numpy as np
import pandas as pd
import utils
from model import  keras_model
import os
import cv2
from keras.utils import np_utils, print_summary
 # or ‘./test’ depending on for which the CSV is being created
mapping = {"melanoma": 1, "nevus": 2, "seborrheic_keratosis": 3}

def prepare_data(root):
    for directory, subdirectories, files in os.walk(root):
        print ('directory',directory)
        for subdirectory in subdirectories:
             folder = root  + "/" +subdirectory
             print ("folder",folder)
             for files in os.listdir(folder):
                 image = cv2.imread(os.path.join(folder,files))
                 image = utils.preprocessing(image)
                 image = utils.segmenting(image)

                 image = utils.resizing(image)
                 value = image.flatten()
                 value = value.astype(float)
                 label =  mapping[subdirectory]
                 value = np.hstack((label,value))
                 df = pd.DataFrame(value).T
                 df = df.sample(frac=1) # shuffle the dataset

                 with open('train_foo.csv', 'a') as dataset:
                    df.to_csv(dataset, header=False, index=False)



def main():
  # prepare_data('./data')
   dataset = pd.read_csv('train_foo.csv')
   dataset = np.array(dataset)
   np.random.shuffle(dataset)
   X = dataset[:,1:]
   Y = dataset[:,0:1]
   X_train= X[0:2199,:]
   X_train = X_train/255.
   X_test = X[2199:2749,:]
   X_test = X_test/255.
   Y = Y.reshape(Y.shape[0], 1)
   Y_train = Y[0:2199, :]
   Y_train = Y_train.T
   Y_test = Y[2199:2749, :]
   Y_test = Y_test.T
   print("number of training examples = " + str(X_train.shape[0]))
   print("number of test examples = " + str(X_test.shape[0]))
   print("X_train shape: " + str(X_train.shape))
   print("Y_train shape: " + str(Y_train.shape))
   print("X_test shape: " + str(X_test.shape))
   print("Y_test shape: " + str(Y_test.shape))
   image_x = 256
   image_y = 256
   print('trainY',Y_train.shape)
   train_y = np_utils.to_categorical(Y_train)
   test_y = np_utils.to_categorical(Y_test)
   train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])
   test_y = test_y.reshape(test_y.shape[1], test_y.shape[2])
   X_train = X_train.reshape(X_train.shape[0], 256, 256, 1)
   X_test = X_test.reshape(X_test.shape[0], 256, 256, 1)
   print("X_train shape: " + str(X_train.shape))
   print("X_test shape: " + str(X_test.shape))
   model, callbacks_list = keras_model(image_x, image_y)
   print('callbacks',callbacks_list)
   print('train y ',train_y)

   model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=10, batch_size=64,
             callbacks=callbacks_list)
   scores = model.evaluate(X_test, test_y, verbose=0)
   print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
   print_summary(model)

   model.save('emojinator.h5')


main()


