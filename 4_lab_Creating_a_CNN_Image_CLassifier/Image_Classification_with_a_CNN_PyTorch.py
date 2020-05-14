'''
Simulating Feature Extraction

The use of convolutional and pooling layers to extract features can seem a little abstract. It can be helpful to visualize 
the outputs of these layers to better understand how they help identify visual features in the images, reducing the size of 
the feature matrices that are passed to the fully-connected classification network layers at the end of the model.

The following code generates a sample image of a geometric shape, and then simulates the following feature extraction layers 
to simulate convolutional and pooling layers in a CNN:
'''
import numpy as np
from skimage.measure import block_reduce
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageFilter


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
A Simple Example

In this notebook, we'll build a simple example CNN using PyTorch. The example is a classification model that can classify an image as a circle, 
a triangle, or a square.

First, we'll use the function we created previously to generate some image files for our classification model. Run the cell below 
to do that (note that it may take several minutes to run)
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

Creating a CNN with PyTorch

First, let's import the PyTorch libraries we'll need.

> Note: The following pip install commands install the CPU-only version of PyTorch on Linux, which is appropriate for the Azure Notebooks environment. 
For instructions on how to install the PyTorch and TorchVision packages on your own system, see https://pytorch.org/get-started/locally/
'''
# Install PyTorch
!pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
!pip install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl

# Import PyTorch libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Other libraries we'll use
import numpy as np
import os
import matplotlib.pyplot as plt

print("Libraries imported - ready to use PyTorch", torch.__version__)


'''

Load Data

PyTorch includes functions for loading and transforming data. We'll use these to create an iterative loader for training data, 
and a second iterative loader for test data (which we'll use to validate the trained model). The loaders will transform 
the image data into tensors, which are the core data structure used in PyTorch, and normalize them so that the pixel values 
are in a scale with a mean of 0.5 and a standard deviation of 0.5.

Run the following cell to define the data loaders.
'''
# Function to ingest data using training and test loaders
def load_dataset(data_path):
    # Load all of the images
    transformation = transforms.Compose([
        # transform to tensors
        transforms.ToTensor(),
        # Normalize the pixel values (in R, G, and B channels)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load all of the images, transforming them
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transformation
    )
    
    
    # Split into training (70% and testing (30%) datasets)
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    # define a loader for the training data we can iterate through in 50-image batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=False
    )
    
    # define a loader for the testing data we can iterate through in 50-image batches
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=False
    )
        
    return train_loader, test_loader


# Now load the images from the shapes folder
data_path = 'shapes/'

# Get the class names
classes = os.listdir(data_path)
classes.sort()
print(len(classes), 'classes:')
print(classes)

# Get the iterative dataloaders for test and training data
train_loader, test_loader = load_dataset(data_path)


'''

Define a Neural Network Model

In PyTorch, you define a neural network model as a class that is derived from the nn.Module base class. 
Your class must define the layers in your network, and provide a forward method that is used to process data through the layers of the network.
'''
# Create a neural net class
class Net(nn.Module):
    # Constructor
    def __init__(self, num_classes=3):
        super(Net, self).__init__()
        
        # Our images are RGB, so input channels = 3. We'll apply 12 filters in the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        
        # We'll apply max pooling with a kernel size of 2
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # A second convolutional layer takes 12 input channels, and generates 12 outputs
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        
        # A third convolutional layer takes 12 inputs and generates 24 outputs
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        
        # A drop layer deletes 20% of the features to help prevent overfitting
        self.drop = nn.Dropout2d(p=0.2)
        
        # Our 128x128 image tensors will be pooled twice with a kernel size of 2. 128/2/2 is 32.
        # So our feature tensors are now 32 x 32, and we've generated 24 of them, so the array is 32x32x24
        # We need to flatten these and feed them to a fully-connected layer
        # to map them to  the probability for each class
        self.fc = nn.Linear(in_features=32 * 32 * 24, out_features=num_classes)

    def forward(self, x):
        # Use a relu activation function after convolution 1 and pool
        x = F.relu(self.pool(self.conv1(x)))
      
        # Use a relu activation function after convolution 2 and pool
        x = F.relu(self.pool(self.conv2(x)))
        
        # Select some features to drop after the 3rd convolution to prevent overfitting
        x = F.relu(self.drop(self.conv3(x)))
        
        # Only drop the features if this is a training pass
        x = F.dropout(x, training=self.training)
        
        # Flatten
        x = x.view(-1, 32 * 32 * 24)
        # Feed to fully-connected layer to predict class
        x = self.fc(x)
        # Return class probabilities via a softmax function 
        return F.log_softmax(x, dim=1)
    
device = "cpu"
if (torch.cuda.is_available()):
    # if GPU available, use cuda (on a cpu, training will take a considerable length of time!)
    device = "cuda"

# Create an instance of the model class and allocate it to the device
model = Net(num_classes=len(classes)).to(device)

print(model)


'''

Train the Model

Now that we've defined a class for the network, we can train it using the image data.

Training consists of an iterative series of forward passes in which the training data is processed in batches by the layers in the network, 
and the optimizer goes back and adjusts the weights. We'll also use a separate set of test images to test the model at the end of each iteration 
(or epoch) so we can track the performance improvement as the training process progresses.

In the example below, we use 5 epochs to train the model using the batches of images loaded by the data loaders, holding back the data in 
the test data loader for validation. After each epoch, a loss function measures the error (loss) in the model and adjusts 
the weights (which were randomly generated for the first iteration) to try to improve accuracy.

> Note: We're only using 5 epochs to reduce the training time for this simple example. A real-world CNN is usually trained over more epochs than this. 
CNN model training is processor-intensive, so it's recommended to perform this on a system that can leverage GPUs (such as the Data Science Virtual Machine 
in Azure) to reduce training time. This will take a while to complete in Azure Notebooks (in which GPUs are not available) - status will be displayed 
as the training progresses. Feel free to go get some coffee!
'''
def train(model, device, train_loader, optimizer, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # Use the CPU or GPU as appropriate
        data, target = data.to(device), target.to(device)
        
        # Reset the optimizer
        optimizer.zero_grad()
        
        # Push the data forward through the model layers
        output = model(data)
        
        # Get the loss
        loss = loss_criteria(output, target)

        # Keep a running total
        train_loss += loss.item()
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        
        # Print metrics so we see some progress
        print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))
            
    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss
            
            
def test(model, device, test_loader):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            
            # Get the predicted classes for this batch
            output = model(data)
            
            # Calculate the loss for this batch
            test_loss += loss_criteria(output, target).item()
            
            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target==predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # return average loss for the epoch
    return avg_loss
    
    
# Use an "Adam" optimizer to adjust weights
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Specify the loss criteria
loss_criteria = nn.CrossEntropyLoss()

# Track metrics in these arrays
epoch_nums = []
training_loss = []
validation_loss = []

# Train over 5 epochs (in a real scenario, you'd likely use many more)
epochs = 5
print('Training on', device)
for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)


'''

View the Loss History

We tracked average training and validation loss for each epoch. We can plot these to verify that loss reduced as the model was trained, and 
to detect overfitting (which is indicated by a continued drop in training loss after validation loss has levelled out or started to increase.
'''
from matplotlib import pyplot as plt

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
# Pytorch doesn't have a built-in confusion matrix metric, so we'll use SciKit-Learn
from sklearn.metrics import confusion_matrix

# Set the model to evaluate mode
model.eval()

# Get predictions for the test data and convert to numpy arrays for use with SciKit-Learn
print("Getting predictions from test set...")
truelabels = []
predictions = []
for data, target in test_loader:
    for label in target.cpu().data.numpy():
        truelabels.append(label)
    for prediction in model.cpu()(data).data.numpy().argmax(1):
        predictions.append(prediction) 

# Plot the confusion matrix
cm = confusion_matrix(truelabels, predictions)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel("Predicted Shape")
plt.ylabel("True Shape")
plt.show()


'''

Save the Model

Now that we have trained the model, we can save its weights. Then later, we can reload those weights into an instance of the same network 
and use it to predict classes from new images.
'''
# Save the model weights
model_file = 'shape-classifier.pth'
torch.save(model.state_dict(), model_file)
print("Model saved.")

# Delete the existing model variable
del model


'''

Use the Model with New Data

Now that we've trained and evaluated our model, we can use it to predict classes for new images.
'''
# Function to predict the class of an image
def predict_image(classifier, image_array):
   
    # Set the classifer model to evaluation mode
    classifier.eval()
    
    # These are the classes our model can predict
    class_names = ['circle', 'square', 'triangle']
    
    # Apply the same transformations as we did for the training images
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Preprocess the imagees
    image_tensor = torch.stack([transformation(image).float() for image in image_array])

    # Predict the class of each input image
    predictions = classifier(image_tensor)
    
    predicted_classes = []
    # Convert the predictions to a numpy array 
    for prediction in predictions.data.numpy():
        # The prediction for each image is the probability for each class, e.g. [0.8, 0.1, 0.2]
        # So get the index of the highest probability
        class_idx = np.argmax(prediction)
        # And append the corresponding class name to the results
        predicted_classes.append(class_names[class_idx])
    return np.array(predicted_classes)


# Now let's try it with a new image
from random import randint

# Create a new model instance and load the weights
model = Net()
model.load_state_dict(torch.load(model_file))

# Create a random test image
img = create_image ((128,128), classes[randint(0, len(classes)-1)])
plt.imshow(img)

# Create an array of (1) images to match the expected input format
image_array = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]).astype('float32')

predicted_classes = predict_image(model, image_array)
print(predicted_classes[0])
