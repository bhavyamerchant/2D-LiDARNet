#CNN Model

#Load Libraries
library(keras)
library(tensorflow)
library(EBImage)
library(imager)
library(abind)
library(caret)

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
setwd()

#Using the function, load in our LiDAR Dataframe
LiDAR <- load_LiDAR()

batch_size <- 128
num_classes <- 2
epochs <- 100

#Input Image Dimensions
img_rows <- 48
img_cols <- 48

train$n <- NULL
test$n <- NULL

#Convert list into 1d array
train$y <- as.array(train$y)
test$y <- as.array(test$y)

#Data Pre-Processing
x_train <- train$x
y_train <- train$y
x_test <- test$x
y_test <- test$y

#Redefine Dimension of train/test inputs
x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

#Transform RGB values into [0,1] range
x_train <- x_train/255
x_test <- x_test/255

c('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples \n')

#Convert Class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

#Append arrays
x <- abind(x_train, x_test, along = 1)


#Row Bind labels
y <- rbind(y_train, y_test)

#Stratify dataset split
index <- createDataPartition(y[,1], p = .8, list = FALSE)

train_x <- x[index,,,]
test_x <- x[-index,,,]

#For some reason, the "createDatatPartion" removes the 4th dimension from train_x and train_y
#We add the 4th dimension back in here
dim(train_x) <- c(dim(train_x),1)
dim(test_x) <- c(dim(test_x),1)

train_y <- y[index,]
test_y <- y[-index,]


#Initialize the Model
CNNmodel <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = input_shape) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.6) %>%
  layer_flatten()%>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.6) %>%
  layer_dense(units = num_classes, activation = 'softmax')

#Compile the Model
CNNmodel %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adam(lr = .001),
  metrics = c('accuracy')
)


early_stop <- callback_early_stopping(monitor = "val_loss", 
                                      patience = 20)

#Train the Model
CNNmodel %>% fit(
  train_x, train_y,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2,
  callback= list(early_stop)
)

scores <- CNNmodel %>% evaluate(test_x, test_y, verbose = 0)

scores


class_names <- c("Tree", "No Tree")

par(mfrow = c(2,2))

plot(as.cimg(t(x_train[10,,,])),
     main=paste("Label:", class_names[y_train[10]+1]))
plot(as.cimg(t(x_train[20,,,])),
     main=paste("Label:", class_names[y_train[20]+1]))
plot(as.cimg(t(x_train[40,,,])),
     main=paste("Label:", class_names[y_train[40]+1]))
plot(as.cimg(t(x_train[80,,,])),
     main=paste("Label:", class_names[y_train[80]+1]))
