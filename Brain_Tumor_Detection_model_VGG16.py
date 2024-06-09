#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import cv2
import imutils
import matplotlib.pyplot as plt
from os import listdir
import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from keras.applications import VGG16


# In[4]:


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



# In[5]:


import cv2

# Define the target size
TARGET_SIZE = (224, 224)

# Load the image
img = cv2.imread(r'D:\data\Downloads\archive (11)\brain_tumor_dataset\yes\Y1.jpg')

# Resize the image to the target size
resized_img = cv2.resize(img, dsize=TARGET_SIZE, interpolation=cv2.INTER_CUBIC)

# Convert the resized image to grayscale
gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)

# Apply Gaussian blur
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold the image
thresh = cv2.threshold(gray_blur, 45, 255, cv2.THRESH_BINARY)[1]

# Erosion and dilation to remove noise
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

# Find contours in the thresholded image
cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour
c = max(cnts, key=cv2.contourArea)

# Find extreme points
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

# Draw contour on the image
img_contour = cv2.drawContours(resized_img.copy(), [c], -1, (0, 255, 255), 4)

# Draw extreme points on the image
img_points = cv2.circle(img_contour.copy(), extLeft, 8, (0, 0, 255), -1)
img_points = cv2.circle(img_points, extRight, 8, (0, 255, 0), -1)
img_points = cv2.circle(img_points, extTop, 8, (255, 0, 0), -1)
img_points = cv2.circle(img_points, extBot, 8, (255, 255, 0), -1)

# Crop the region of interest
ADD_PIXELS = 0
cropped_img = resized_img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()


# In[6]:


img_cnt = cv2.drawContours(img.copy(), [c], -1, (0, 255, 255), 4)
plt.figure(figsize=(15, 6))
plt.subplot(141)
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.title('Step 1. Get the original image')
plt.subplot(142)
plt.imshow(img_cnt)
plt.xticks([])
plt.yticks([])
plt.title('Step 2. Find the biggest contour')
plt.subplot(143)
plt.imshow(img_points)
plt.xticks([])
plt.yticks([])
plt.title('Step 3. Find the extreme points')
plt.subplot(144)
plt.imshow(cropped_img)
plt.xticks([])
plt.yticks([])
plt.title('Step 4. Crop the image')
plt.show()


# In[4]:


# Images size  and no of images before Augmentation

#  no of images before Data Augmentation

yes_images_dir = r'D:\data\Downloads\archive (10)\brain_tumor_dataset\yes'
no_images_dir = r'D:\data\Downloads\archive (10)\brain_tumor_dataset\no'

# Count the number of images in the "yes" directory
yes_image_count = len(os.listdir(yes_images_dir))

# Count the number of images in the "no" directory
no_image_count = len(os.listdir(no_images_dir))

print(f"Number of images in 'yes' directory: {yes_image_count}")
print(f"Number of images in 'no' directory: {no_image_count}")


# In[5]:


#Data augmenation on images to generate more of samples of images 
def augment_data(file_dir, n_generated_samples, save_to_dir):
   

   
   data_gen = ImageDataGenerator(rotation_range=10, 
                                 width_shift_range=0.1, 
                                 height_shift_range=0.1, 
                                 shear_range=0.1, 
                                 brightness_range=(0.3, 1.0),
                                 horizontal_flip=True, 
                                 vertical_flip=True,
                                 color='rgb',
                                 fill_mode='nearest'
                                )

   
   for filename in listdir(file_dir):
       # load the image
       image = cv2.imread(file_dir + '\\' + filename)
       # reshape the image
       image = image.reshape((1,)+image.shape)
       # prefix of the names for the generated sampels.
       save_prefix = 'aug_' + filename[:-4]
       # generate 'n_generated_samples' sample images
       i=0
       for batch in data_gen.flow(x=image, batch_size=1, save_to_dir=save_to_dir, 
                                          save_prefix=save_prefix, save_format='jpg'):
           i += 1
           if i > n_generated_samples:
               break


# In[ ]:


augmented_data_path = 'D:/data/Downloads/archive (10)/brain_tumor_dataset/data_augmented_path/'

# augment data for the examples with label equal to 'yes' representing tumorous examples
augment_data(file_dir=r"D:\data\Downloads\archive (10)\brain_tumor_dataset\yes", n_generated_samples=6, save_to_dir=os.path.join(augmented_data_path, 'aug_yes'))

# augment data for the examples with label equal to 'no' representing non-tumorous examples
augment_data(file_dir=r"D:\data\Downloads\archive (10)\brain_tumor_dataset\no", n_generated_samples=9, save_to_dir=os.path.join(augmented_data_path, 'aug_no'))


# In[6]:


def count_augmented_images(augmented_data_path):
    augmented_yes_path = os.path.join(augmented_data_path, 'aug_yes')
    augmented_no_path = os.path.join(augmented_data_path, 'aug_no')

    num_augmented_yes_images = len(os.listdir(augmented_yes_path))
    num_augmented_no_images = len(os.listdir(augmented_no_path))

    print(f"Number of augmented 'yes' images: {num_augmented_yes_images}")
    print(f"Number of augmented 'no' images: {num_augmented_no_images}")

# Example usage:
augmented_data_path = r'D:\data\Downloads\archive (10)\brain_tumor_dataset\data_augmented_path'
count_augmented_images(augmented_data_path)


# In[7]:


def data_summary(main_path):
    aug_yes_path = os.path.join(main_path, 'data_augmented_path', 'aug_yes')
    aug_no_path = os.path.join(main_path, 'data_augmented_path', 'aug_no')
    
    n_pos = len(os.listdir(aug_yes_path))
    n_neg = len(os.listdir(aug_no_path))
    m = n_pos + n_neg
    
    print(f"Total number of samples: {m}")
    print(f"Number of positive samples: {n_pos}")
    print(f"Number of negative samples: {n_neg}")

# Example usage:
main_path = r"D:\data\Downloads\archive (10)\brain_tumor_dataset"
data_summary(main_path)


# In[8]:


from sklearn.utils import shuffle
import numpy as np
def load_data(dir_list, image_size):


    # load all images in a directory
    X = []
    y = []
    image_width, image_height = image_size
    
    for directory in dir_list:
        for filename in listdir(directory):
            # load the image
            image = cv2.imread(directory + '\\' + filename)
            # crop the brain and ignore the unnecessary rest part of the image
            image = crop_brain_contour(image, plot=False)
            # resize image
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            # normalize values
            image = image / 255.
            # convert image to numpy array and append it to X
            X.append(image)
            # append a value of 1 to the target array if the image
            # is in the folder named 'yes', otherwise append 0.
            if directory[-3:] == 'yes':
                y.append([1])
            else:
                y.append([0])
                
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle the data
    X, y = shuffle(X, y)
    
    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')
    return X,y


# In[9]:


data_augmented_path="D:\data\Downloads\archive (10)\brain_tumor_dataset\data_augmented_path"
aug_yes=r"D:\data\Downloads\archive (10)\brain_tumor_dataset\data_augmented_path\aug_yes"
aug_no=r"D:\data\Downloads\archive (10)\brain_tumor_dataset\data_augmented_path\aug_no"

image_width,image_height=(224,224)
X,y=load_data([aug_yes,aug_no],(image_width,image_height))


# In[6]:


# Plot Sample Images 

def plot_sample_images(X,y,image_size=(224, 224),n=50):
    for label in [0,1]:
        images=X[np.argwhere(y==label)]
        n_images=images[:n]
        columns_n=10
        rows_n=int(n/columns_n)
        i=1
        plt.figure(figsize=(20,10))
        for image in n_images:
            plt.subplot(rows_n,columns_n,i)
            plt.imshow(image[0])
            #i += 1
            plt.tick_params(axis='both', which='both', 
                top=False, bottom=False, left=False, right=False,
                labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            i +=1
        labels=lambda label:"Yes" if label==1 else "No"
        plt.suptitle(f"Brain Tumor: {labels(label)}")
            
        plt.show()


# In[7]:


plot_sample_images(X,y)


# In[96]:


import os

base_dir=r"D:\data\Downloads\archive (10)\brain_tumor_dataset\tumor _and_nontumor"
if not os.path.isdir(base_dir):
    os.mkdir(base_dir)


# In[121]:


# Create subdirectories within the base directory
train_dir = os.path.join(base_dir ,'train')
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)

test_dir = os.path.join(base_dir ,'test')
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

validation_dir = os.path.join(base_dir ,'validation')
if not os.path.isdir(validation_dir):
    os.mkdir(validation_dir)
    


# In[122]:


import os

# Define paths
train_dir = r"D:\data\Downloads\archive (10)\brain_tumor_dataset\tumor_and_nontumor\train\tumor"
test_dir = r"D:\data\Downloads\archive (10)\brain_tumor_dataset\tumor_and_nontumor\test\tumor"
validation_dir = r"D:\data\Downloads\archive (10)\brain_tumor_dataset\tumor_and_nontumor\validation\tumor"

# Check and create directories
if not os.path.isdir(train_dir):
    os.makedirs(train_dir)  # Use makedirs to create parent directories if they don't exist
    os.path.join(train_dir,'tumor')
if not os.path.isdir(test_dir):
    os.makedirs(test_dir)
    os.path.join(test_dir,'tumor')
if not os.path.isdir(validation_dir):
    os.makedirs(validation_dir)
    os.path.join(test_dir,'tumor')


# In[118]:


import os

train_dir=r"D:\data\Downloads\archive (10)\brain_tumor_dataset\tumor _and_nontumor\train\nontumor"
test_dir=r"D:\data\Downloads\archive (10)\brain_tumor_dataset\tumor _and_nontumor\test\nontumor"
validation_dir=r"D:\data\Downloads\archive (10)\brain_tumor_dataset\tumor _and_nontumor\validation\nontumor"

# Check and create directories
if not os.path.isdir(train_dir):
    os.makedirs(train_dir)  # Use makedirs to create parent directories if they don't exist
    os.pathjoin(train_dir,'nontumor')
if not os.path.isdir(test_dir):
    os.makedirs(test_dir)
    os.path.join(test_dir,'nontumor')
if not os.path.isdir(validation_dir):
    os.makedirs(validation_dir)
    os.path.join(validation_dir,'tumor')


# In[11]:


# splitting the data into train, test,validate 
def split_data(X, y, test_size=0.2):
       
    
    
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# In[12]:


X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)


# In[13]:


print ("number of training images = " + str(X_train.shape[0]))
print ("number of test images = " + str(X_test.shape[0]))
print ("number of validation images = " + str(X_val.shape[0]))


# In[14]:


print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(y_train.shape))
print ("X_val  shape: " + str(X_val.shape))
print ("Y_val  shape: " + str(y_val.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(y_test.shape))


# In[15]:


# model Building

NUM_CLASSES = 1

# Load the VGG16 model with the top (classification) layers included
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the weights of the VGG16 layers
for layer in vgg.layers:
    layer.trainable = False

# Create a Sequential model and add the VGG16 base
vgg16 = Sequential()
vgg16.add(vgg)

# Add additional layers on top of VGG16
vgg16.add(Dropout(0.3))
vgg16.add(Flatten())
vgg16.add(Dropout(0.5))
vgg16.add(Dense(NUM_CLASSES, activation='sigmoid'))


optimizer = RMSprop(lr=1e-4)

# Compile the model with the defined optimizer
vgg16.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)


# In[16]:


# Display model summary
vgg16.summary()


# In[17]:


# unique file name that will include the epoch and the validation (development) accuracy
from keras.callbacks import ModelCheckpoint
import time
filepath="cnn-parameters-improvement-{epoch:02d}-{val_acc:.2f}"
# save the model with the best validation (development) accuracy till now
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))


# In[18]:


history=vgg16.fit(X_train,y_train,batch_size=32,epochs=5,validation_data=(X_val,y_val,verbose=1))


# In[19]:


history=vgg16.history.history


# In[20]:


for key in history.keys():
    print(key)
    


# In[25]:


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
    plt.show()


# In[26]:


plot_metrics(history) 


# In[30]:


vgg16.metrics_names


# In[32]:


loss, acc = vgg16.evaluate(x=X_test, y=y_test)


# In[33]:


#Accuracy of the best model on the testing data#
print (f"Test Loss = {loss}")
print (f"Test Accuracy = {acc}")


# In[38]:


# F1 Score
from sklearn.metrics import f1_score
def compute_f1_score(y_true, prob):
    # convert the vector of probabilities to a target vector
    y_pred = np.where(prob > 0.5, 1, 0)
    
    score = f1_score(y_true, y_pred)
    
    return score


# In[41]:


y_test_prob= vgg16.predict(X_test)


# In[42]:


f1score=compute_f1_score(y_test,y_test_prob)
print(f'F1 score:{f1score}')


# In[49]:


y_pred = np.where(y_test_prob > 0.5, 1, 0)

precision = precision_score(y_test, y_pred)
print(f'Precision: {precision}')


# In[61]:


y_test_pred = np.where(y_test_prob > 0.5, 1, 0)
recall = recall_score(y_test, y_test_pred)

print('Recall:', recall)


# In[56]:


# Results Interpreatation 

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
data_percentage(y)


# In[57]:


print("Training Data:")
data_percentage(y_train)
print("Validation Data:")
data_percentage(y_val)
print("Testing Data:")
data_percentage(y_test)


# In[ ]:




