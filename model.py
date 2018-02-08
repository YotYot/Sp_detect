from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import keras
import numpy as np
from scipy import misc
import glob
import matplotlib.pyplot as plt
import os
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import time

im_array = []
tag_array = []

cmap = plt.get_cmap('jet')
for pic in glob.glob('/Users/yotam/OneDrive - Cadence Design Systems Inc/Private/final project/submission/runme/small_bw_db/img/*.jpg'):
    img = misc.imread(pic)
    img_3l = [img,img,img]
#    rgba_img = cmap(img)
#    rgb_img = np.delete(rgba_img, 3, 2)
    img_3l = np.array(img_3l)
    rgb_img = img_3l.transpose()
    if (rgb_img.shape == (200,200,3)):
        im_array.append(rgb_img)
        filename = os.path.basename(pic)
        filename_no_ext = os.path.splitext(filename)[0]
        class_filename = '/Users/yotam/OneDrive - Cadence Design Systems Inc/Private/final project/submission/runme/untitled folder/Archive/'+filename_no_ext+'.mat.txt'
        with open(class_filename,'r') as f:
            class_int = int(f.read())
            b = np.zeros(3)
            b[class_int-1] = 1
            tag_array.append(b)

#plt.imshow(np.uint8((im_array[0])))
#image = misc.imread('/Users/yotam/OneDrive - Cadence Design Systems Inc/Private/final project/submission/runme/small_bw_db/img/102F1_C1_R150.jpg')
#Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()
for layer in model_vgg16_conv.layers:
    layer.trainable = False
#Create your own input format (here 3x200x200)
input = Input(shape=(200,200,3),name = 'image_input')

#Use the generated model
output_vgg16_conv = model_vgg16_conv(input)

#Add the fully-connected layers
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(3, activation='softmax', name='predictions')(x)

#Create your own model
my_model = Model(input=input, output=x)

#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
my_model.summary()
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
curr_time = time.strftime("%m_%d_%H_%M_%S", time.gmtime())
checkpointer = ModelCheckpoint(filepath='./weights_'+curr_time+'/weights.hdf5', verbose=1, save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs_'+curr_time, histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

my_model.compile(loss='mean_squared_error', optimizer='sgd',metrics=['accuracy'])
im_array = np.array(im_array)
tag_array = np.array(tag_array)
my_model.fit(im_array,tag_array,validation_split=0.3,verbose=1,epochs=10,callbacks=[checkpointer,tensorboard])