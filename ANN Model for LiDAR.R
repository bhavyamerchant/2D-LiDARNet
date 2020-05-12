#ANN for LiDAR PRoject

#Load Relelvant Libraries
library(keras)
library(imager)
library(tidyr)
library(ggplot2)

##Function to read in mNIST (ubyte) like data
load_LiDAR <- function()
{
  load_image_file <- function(filename) 
  {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) 
  {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  train <<- load_image_file('./train-images-idx3-ubyte')
  test <<- load_image_file('./test-images-idx3-ubyte')
  
  train$y <<- load_label_file('./train-labels-idx1-ubyte')
  test$y <<- load_label_file('./test-labels-idx1-ubyte') 
}

getwd()
setwd() #Make sure to change the WD from 28x28 to 48x48 or vice versa

#Using the function, load in our LiDAR Dataframe
LiDAR <- load_LiDAR()

#Remove first "n" column
train$n <- NULL
test$n <- NULL

#Convet the 2D matricies into a 3D arrays with 48x48/28x28 matrices
dim(train$x) <- c(688, 28, 28)
dim(test$x) <- c(173, 28, 28)

#Convert labels into 1d array
train$y <- as.array(train$y)
test$y <- as.array(test$y)


#Initializing variables
train_images <- train$x
train_labels <- train$y
test_images <- test$x
test_labels <- test$y

#Contain Pixel densities from 0 to 1
train_images <- train_images/255
test_images <- test_images/255

#One-HOt encode the labels (0 is No Tree, 1 is Tree)
train_labels_onehot <- to_categorical(train_labels)
test_labels_onehot <- to_categorical(test_labels, 
                                     num_classes = ncol(train_labels_onehot))

use_session_with_seed(1)   # SETTING THE SEED ALWAYS


#Early-Stop to prevent overfitting
early_stop <- callback_early_stopping(monitor = "val_loss", 
                                      patience = 20)

#Initialize the ANN model
ANNmodel <- keras_model_sequential(layers=list(
  layer_flatten(input_shape = c(28, 28)),
  layer_dense(units = 200, activation = 'relu'),
  layer_dense(units = 2, activation = 'softmax')))

ANNmodel

compile(ANNmodel,
        optimizer = 'adam', 
        loss='categorical_crossentropy',
        metrics = 'accuracy')

history <- fit(ANNmodel, 
               train_images, 
               train_labels_onehot,
               validation_split = 0.2, batch_size=128,
               epochs = 200,
               callbacks = list(early_stop))

history

plot(history, smooth=F)



evaluate(ANNmodel, test_images, test_labels_onehot)

#use_session_with_seed(1)



par(mfrow = c(2,2))
class_names <- c("No Tree", "Tree" )
plot(as.cimg(t(train_images[84, ,])),
     main=paste("Label:", class_names[train_labels[84]+1]))
plot(as.cimg(t(train_images[200, ,])),
     main=paste("Label:", class_names[train_labels[200]+1]))
plot(as.cimg(t(train_images[531, ,])),
     main=paste("Label:", class_names[train_labels[531]+1]))
plot(as.cimg(t(train_images[633, ,])),
     main=paste("Label:", class_names[train_labels[633]+1]))
