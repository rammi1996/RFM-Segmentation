#!/usr/bin/env python
# coding: utf-8

# In[35]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import cv2
import imutils
import matplotlib.pyplot as plt
from os import listdir
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization,  Flatten, Dense, MaxPooling2D
from tensorflow.keras.models import Sequential, Model




def crop_brain_contour(image, plot=False):
    
    import imutils
    import cv2
    from matplotlib import pyplot as plt
    
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            

    if plot:
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        
        plt.tick_params(axis='both', which='both', 
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        
        plt.title(' Image')
            
        plt.subplot(1, 2, 2)
        plt.imshow(new_image)

        plt.tick_params(axis='both', which='both', 
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.title('Cropped Image')
        
        plt.show()
    
    return new_image

yes_images_dir = r'D:\data\Downloads\archive (10)\brain_tumor_dataset\yes'
no_images_dir = r'D:\data\Downloads\archive (10)\brain_tumor_dataset\no'
ex_img = cv2.imread(f'{yes_images_dir}/Y1.jpg')
ex_new_img = crop_brain_contour(ex_img, True)

ex_img = cv2.imread(f'{no_images_dir}/no 1.jpg')
ex_new_img = crop_brain_contour(ex_img, True)





image_width,image_height=(240,240)




# Data Splitting
# Train
# Test
# validation


def split_data(X, y, test_size=0.2):
       
    
    
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)




print ("number of training images = " + str(X_train.shape[0]))
print ("number of test images = " + str(X_test.shape[0]))
print ("number of validation images = " + str(X_val.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(y_train.shape))
print ("X_val  shape: " + str(X_val.shape))
print ("Y_val  shape: " + str(y_val.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(y_test.shape))


def build_model(input_shape):
    model = Sequential()
    model.add(Input(input_shape))
    model.add(ZeroPadding2D((2, 2)))
    model.add(Conv2D(32, (7, 7), strides=(1, 1), name='conv0',activation='relu'))
    model.add(BatchNormalization(axis=3, name='bn0'))
    model.add(MaxPooling2D((4, 4), name='max_pool0'))
    model.add(MaxPooling2D((4, 4), name='max_pool1'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid', name='fc'))
    
    return model

# Define the input shape
input_shape = (240, 240, 3)

# Build the model
model = build_model(input_shape)

# complie the model

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

# Train the model

history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_val, y_val),
    verbose=1
)

history=model.history.history
for key in history.keys():
    print(key)
    
# plot loss and accuracy

def plot_metrics(history):
    loss = history['loss']
    accuracy = history['accuracy']
    val_loss = history['val_loss']
    val_accuracy = history['val_accuracy']
    
    epochs = range(1, len(loss) + 1)
    
    # Plot Training and Validation Loss
    plt.plot(epochs, loss,  label='Training loss')
    plt.plot(epochs, val_loss,  label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot Training and Validation Accuracy
    plt.figure()
    plt.plot(epochs, accuracy,  label='Training accuracy')
    plt.plot(epochs, val_accuracy,  label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show())
plot_metrics(history)



loss, acc = model.evaluate(x=X_test, y=y_test)



print (f"Test Loss = {loss}")
print (f"Test Accuracy = {acc}")



# F1 Score
from sklearn.metrics import f1_score
def compute_f1_score(y_true, prob):
    # convert the vector of probabilities to a target vector
    y_pred = np.where(prob > 0.5, 1, 0)
    
    score = f1_score(y_true, y_pred)
    
    return score


y_test_prob= model.predict(X_test)



f1score=compute_f1_score(y_test,y_test_prob)
print(f'F1 score:{f1score}')


y_val_prob = model.predict(X_val)
f1score_val = compute_f1_score(y_val, y_val_prob)
print(f"F1 score: {f1score_val}")



precision = precision_score(y_test, y_pred)
print(f'Precision: {precision}')
y_pred = np.where(y_test_prob > 0.5, 1, 0)





y_test_pred = np.where(y_test_prob > 0.5, 1, 0)
recall = recall_score(y_test, y_test_pred)

print('Recall:', recall)


def data_percentage(y):
    m = len(y)
    n_positive = np.sum(y)
    n_negative = m - n_positive

    pos_prec = (n_positive / m) * 100
    neg_prec = (n_negative / m) * 100

    print(f"Number of examples: {m}")
    print(f"Percentage of positive examples: {pos_prec:.2f}%, number of pos examples: {n_positive}") 
    print(f"Percentage of negative examples: {neg_prec:.2f}%, number of neg examples: {n_negative}") 

# Example usage:
# Assuming 'y' contains the target labels
def data_percentage(y):
    m = len(y)
    n_positive = np.sum(y)
    n_negative = m - n_positive

    pos_prec = (n_positive / m) * 100
    neg_prec = (n_negative / m) * 100

    print(f"Number of examples: {m}")
    print(f"Percentage of positive examples: {pos_prec:.2f}%, number of pos examples: {n_positive}") 
    print(f"Percentage of negative examples: {neg_prec:.2f}%, number of neg examples: {n_negative}") 



# print training test validation
print("Training Data:")
data_percentage(y_train)
print("Validation Data:")
data_percentage(y_val)
print("Testing Data:")
data_percentage(y_test)



# Save the trained model

model.save("Brain_Tumor_detection_model_CNN")


# In[42]:


from sklearn.metrics import precision_score, recall_score, f1_score

# Assuming y_true contains the true labels and y_pred contains the predicted labels
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1score}')


