'''
Simulating Feature Extraction

The use of convolutional and pooling layers to extract features can seem a little abstract. It can be helpful to visualize the outputs of these layers to better understand how they help identify visual features in the images, reducing the size of the feature matrices that are passed to the fully-connected classification network layers at the end of the model.

The following code generates a sample image of a geometric shape, and then simulates the following feature extraction layers to simulate convolutional and pooling layers in a CNN:
'''
import numpy as np
from skimage.measure import block_reduce
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageFilter
%matplotlib inline


# Function to create a random image (of a square, circle, or triangle)
def create_image (size, shape):
    from random import randint
    import numpy as np
    from PIL import Image, ImageDraw
    
    xy1 = randint(10,40)
    xy2 = randint(60,100)
    col = (randint(0,200), randint(0,200), randint(0,200))

    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    if shape == 'circle':
        draw.ellipse([(xy1,xy1), (xy2,xy2)], fill=col)
    elif shape == 'triangle':
        draw.polygon([(xy1,xy1), (xy2,xy2), (xy2,xy1)], fill=col)
    else: # square
        draw.rectangle([(xy1,xy1), (xy2,xy2)], fill=col)
    del draw
    
    return np.array(img)

# Create a 128 x 128 pixel image (Let's use a square)
img = Image.fromarray(create_image((128,128), 'square'))

# Now let's generate some feature extraction layers
layers = []

# Define filter kernels - we'll use two filters
kernel_size = (3,3) # kernels are 3 x 3
kernel_1 = (1, 0, -100,
            1, 0, -100,
            1, 0, -100) # Mask for first filter
kernel_2 = (-200, 0, 0,
            0, -200, 0,
            -0, 0, -200) # Mask for second filter

# Define kernel size for pooling
pool_size = (2,2,1) # Pool filter is 2 x 2 pixels for each channel (so image size will half with each pool)

# Convolutional layer
this_layer = []
# Apply each filter to the original image - this generates a layer with two filtered images
this_layer.append(np.array(img.filter(ImageFilter.Kernel(kernel_size, kernel_1))))
this_layer.append(np.array(img.filter(ImageFilter.Kernel(kernel_size, kernel_2))))
layers.append(this_layer)
             
# Add a Pooling layer - pool each image in the previous layer by only using the maximum value in each 2x2 area
this_layer = []
for i in layers[len(layers)-1]:
    # np.maximum implements a ReLU activation function so all pixel values are >=0
    this_layer.append(np.maximum(block_reduce(i, pool_size, np.max), 0))
layers.append(this_layer)

# Add a second convolutional layer - generates a new layer with 4 images (2 filters applied to 2 images in the previous layer)
this_layer = []
for i in layers[len(layers)-1]:
    this_layer.append(np.array(Image.fromarray(i).filter(ImageFilter.Kernel(kernel_size, kernel_1))))
    this_layer.append(np.array(Image.fromarray(i).filter(ImageFilter.Kernel(kernel_size, kernel_2))))
layers.append(this_layer)

# Add a second Pooling layer - pool each image in the previous layer
this_layer = []
for i in layers[len(layers)-1]:
    # np.maximum implements a ReLU activation function so all pixel values are >=0
    this_layer.append(np.maximum(block_reduce(i, pool_size, np.max), 0))
layers.append(this_layer)

# Set up a grid to plot the images in each layer
fig = plt.figure(figsize=(16, 24))
rows = len(layers) + 1
columns = len(layers[len(layers)-1])
row = 0
image_no = 1

# Plot the original image as layer 1
a=fig.add_subplot(rows,columns,image_no)
imgplot = plt.imshow(img)
a.set_title('Original')

# Plot the convolved and pooled layers
for layer in layers:
    row += 1
    image_no = row * columns
    for image in layer:
        image_no += 1
        a=fig.add_subplot(rows,columns,image_no)
        imgplot = plt.imshow(image)
        a.set_title('Layer ' + str(row))
plt.show() 


'''


Look at the output, and note the following:

    Each convolutional layer applies two filters, extracting features such as edges.
    Each pooling layer reduces the overall size of the image by half, and has the effect of exaggerating the features by taking the maximum pixel value in each 2 x 2 square area.
    A ReLU activation function is applied to each pooling layer, ensuring that any negative pixel values are set to 0.
    The final layer represents the feature maps that have been extracted from the images - you should be able to see that the straight edges of the square have been detected.

The feature maps are just arrays of numbers, as shown by the following code:
'''
for arr in layers[len(layers)-1]:
    print(arr.shape, ':', arr.dtype)
    print (arr)



'''
Building a CNN

There are several commonly used frameworks for creating CNNs, including PyTorch, Tensorflow, the Microsoft Cognitive Toolkit (CNTK), 
and Keras (which is a high-level API that can use Tensorflow or CNTK as a back end).
A Simple Example

The example is a classification model that can classify images of geometric shapes.

First, we'll use the function we created previously to generate some images for our classification model. Run the cell below to do that 
(note that it may take several minutes to run)
'''
# function to create a dataset of images
def generate_image_data (classes, size, cases, img_dir):
    import os, shutil
    from PIL import Image
    
    if os.path.exists(img_dir):
        replace_folder = input("Image folder already exists. Enter Y to replace it (this can take a while!). \n")
        if replace_folder == "Y":
            print("Deleting old images...")
            shutil.rmtree(img_dir)
        else:
            return # Quit - no need to replace existing images
    os.makedirs(img_dir)
    print("Generating new images...")
    i = 0
    while(i < (cases - 1) / len(classes)):
        if (i%25 == 0):
            print("Progress:{:.0%}".format((i*len(classes))/cases))
        i += 1
        for classname in classes:
            img = Image.fromarray(create_image(size, classname))
            saveFolder = os.path.join(img_dir,classname)
            if not os.path.exists(saveFolder):
                os.makedirs(saveFolder)
            imgFileName = os.path.join(saveFolder, classname + str(i) + '.jpg')
            try:
                img.save(imgFileName)
            except:
                try:
                    # Retry (resource constraints in Azure notebooks can cause occassional disk access errors)
                    img.save(imgFileName)
                except:
                    # We gave it a shot - time to move on with our lives
                    print("Error saving image", imgFileName)
            
# Our classes will be circles, squares, and triangles
classnames = ['circle', 'square', 'triangle']

# All images will be 128x128 pixels
img_size = (128,128)

# We'll store the images in a folder named 'shapes'
folder_name = 'shapes'

# Generate 1200 random images.
generate_image_data(classnames, img_size, 1200, folder_name)

print("Image files ready in %s folder!" % folder_name)


'''

Setting up the Frameworks

Now that we have our data, we're ready to build a CNN. The first step is to install and configure the framework we want to use.

We're going to use the Keras machine learning framewor with the default TensorFlow back-end.

> Note: To install Keras on your own system, consult the Keras installation documentation at https://keras.io/#installation.
'''
!pip install --upgrade keras

import keras
from keras import backend as K

print('Keras version:',keras.__version__)


'''

Preparing the Data

Before we can train the model, we need to prepare the data. We'll divide the feature values by 255 to normalize them as floating point values 
between 0 and 1, and we'll split the data so that we can use 70% of it to train the model, and hold back 30% to validate it. When loading the data, 
the data generator will assing "hot-encoded" numeric labels to indicate which class each image belongs to based on the subfolders in which the data is stored. 
In this case, there are three subfolders - circle, square, and triangle, so the labels will consist of three 0 or 1 values indicating which of these classes 
is associated with the image - for example the label [0 1 0] indicates that the image belongs to the second class (square).
'''
from keras.preprocessing.image import ImageDataGenerator

data_folder = 'shapes'
img_size = (128, 128)
batch_size = 30

print("Getting Data...")
datagen = ImageDataGenerator(rescale=1./255, # normalize pixel values
                             validation_split=0.3) # hold back 30% of the images for validation

print("Preparing training dataset...")
train_generator = datagen.flow_from_directory(
    data_folder,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data

print("Preparing validation dataset...")
validation_generator = datagen.flow_from_directory(
    data_folder,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data

classnames = list(train_generator.class_indices.keys())
print("class names: ", classnames)


'''

Defining the CNN

Now we're ready to train our model. This involves defining the layers for our CNN, and compiling them for multi-class classification.
'''
# Define a CNN classifier network
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers

# Define the model as a sequence of layers
model = Sequential()

# The input layer accepts an image and applies a convolution that uses 32 6x6 filters and a rectified linear unit activation function
model.add(Conv2D(32, (6, 6), input_shape=train_generator.image_shape, activation='relu'))

# Next we;ll add a max pooling layer with a 2x2 patch
model.add(MaxPooling2D(pool_size=(2,2)))

# We can add as many layers as we think necessary - here we'll add another convolution, max pooling, and dropout layer
model.add(Conv2D(32, (6, 6), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# And another set
model.add(Conv2D(32, (6, 6), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# A dropout layer randomly drops some nodes to reduce inter-dependencies (which can cause over-fitting)
model.add(Dropout(0.2))

# Now we'll flatten the feature maps and generate an output layer with a predicted probability for each class
model.add(Flatten())
model.add(Dense(train_generator.num_classes, activation='softmax'))

# We'll use the ADAM optimizer
# For information about the optimizers available in PyTorch, see https://pytorch.org/docs/stable/optim.html#algorithms
opt = optimizers.Adam(lr=0.001)

# With the layers defined, we can now compile the model for categorical (multi-class) classification
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

print(model.summary())



'''

Training the Model

With the layers of the CNN defined, we're ready to train the model using our image data. In the example below, we use 3 iterations (epochs) to train 
the model in 30-image batches, holding back 30% of the data for validation. After each epoch, the loss function measures the error (loss) in the model 
and adjusts the weights (which were randomly generated for the first iteration) to try to improve accuracy.

> Note: We're only using 3 epochs to reduce the training time for this simple example. A real-world CNN is usually trained over more epochs than this. 
CNN model training is processor-intensive, so it's recommended to perform this on a system that can leverage GPUs (such as the Data Science 
Virtual Machine in Azure) to reduce training time. This will take a while to complete in Azure Notebooks (in which GPUs are not available) - status 
will be displayed as the training progresses. Feel free to go get some coffee!
'''
# Train the model over 3 epochs using the validation holdout dataset for validation
num_epochs = 3
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = num_epochs)


'''

View the Loss History

We tracked average training and validation loss history for each epoch. We can plot these to verify that loss reduced as the model was trained, 
and to detect overfitting (which is indicated by a continued drop in training loss after validation loss has levelled out or started to increase.
'''
from matplotlib import pyplot as plt

epoch_nums = range(1,num_epochs+1)
training_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()


'''

Evaluate Model Performance

We can see the final accuracy based on the test data, but typically we'll want to explore performance metrics in a little more depth. 
Let's plot a confusion matrix to see how well the model is predicting each class.
'''
#Keras doesn't have a built-in confusion matrix metric, so we'll use SciKit-Learn
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

print("Generating predictions from validation data...")
# Get the first batch of validation data
x_test = validation_generator[0][0]
y_test = validation_generator[0][1]
# iterate through the remaining batches, appending them
batches = range(1,len(validation_generator))
for b in batches:
    x_test = np.append(x_test, validation_generator[b][0], 0)
    y_test = np.append(y_test, validation_generator[b][1], 0)

# Use the model to predict the class
class_probabilities = model.predict(x_test)

# The model returns a probability value for each class
# The one with the highest probability is the predicted class
predictions = np.argmax(class_probabilities, axis=1)

# The actual labels are hot encoded (e.g. [0 1 0], so get the one with the value 1
true_labels = np.argmax(y_test, axis=1)

# Plot the confusion matrix
cm = confusion_matrix(true_labels, predictions)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classnames))
plt.xticks(tick_marks, classnames, rotation=85)
plt.yticks(tick_marks, classnames)
plt.xlabel("Predicted Shape")
plt.ylabel("True Shape")
plt.show()


'''

Save the Model

Now that we have trained the model, we can save it with the trained weights. Then later, we can reload it and use it to predict classes from new images.
'''
from keras.models import load_model

modelFileName = 'shape-classifier.h5'

model.save(modelFileName) # saves the trained model
print("Model saved.")

del model  # deletes the existing model variable


'''

Using the Trained Model

Now that we've trained the model, we can use it to predict the class of a new image.
'''
# Function to predict the class of an image
def predict_image(classifier, image_array):
    import numpy as np
    
    # We need to format the input to match the training data
    # The generator loaded the values as floating point numbers
    # and normalized the pixel values, so...
    imgfeatures = image_array.astype('float32')
    imgfeatures /= 255
    
    # These are the classes our model can predict
    classes = ['circle', 'square', 'triangle']
    
    # Predict the class of each input image
    predictions = classifier.predict(imgfeatures)
    
    predicted_classes = []
    for prediction in predictions:
        # The prediction for each image is the probability for each class, e.g. [0.8, 0.1, 0.2]
        # So get the index of the highest probability
        class_idx = np.argmax(prediction)
        # And append the corresponding class name to the results
        predicted_classes.append(classes[int(class_idx)])
    # Return the predictions as a JSON
    return predicted_classes


from random import randint
import numpy as np

# load the saved model
model = load_model(modelFileName) 

# Create a random test image
img = create_image ((128,128), classnames[randint(0, len(classnames)-1)])
plt.imshow(img)

# Create an array of (1) images to match the expected input format
img_array = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

# get the predicted clases
predicted_classes = predict_image(model, img_array)

# Display the prediction for the first image (we only submitted one!)
print(predicted_classes[0])