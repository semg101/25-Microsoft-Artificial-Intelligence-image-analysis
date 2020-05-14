#Run the following cell to see the results:
import numpy as np
import random
from scipy.misc import derivative

# Known values for input (feature) and output (label)
X = 2.1
Y = 1

# By how much should we adjust the weight with each iteration
LR = 1

# Neuron function
def neuron(x, w):
    from scipy.special import expit as sigmoid
    return sigmoid(x * w)

# Function to calculate loss
def lossfunc(w):
    return abs(Y - neuron(X, w)**2)

# Initialize weight with a random value between 0 and 1
w = random.uniform(0,1)

# Call the function over 5 iterations (epochs), updating the weight and recording the loss each time
e = 1
weights = []
losses = []
while e < 6:
    print('Epoch:', e)
    e += 1
    weights.append(w)
    print('\tWeight:%.20f' % w)

    # Pass the value and weight forward through the neuron
    y = neuron(X, w)
    print('\tTrue value:%.20f' % Y)
    print('\tOutput value:%.20f' % y)

    # Calculate loss
    loss = lossfunc(w)
    losses.append(loss)
    print('\tLoss: %.20f' % loss)

    # Which way should we adjust w to reduce loss?
    dw = derivative(lossfunc, w)
    print('\tDerivative:%.20f' % dw)

    if dw > 0:
        # Slope is positive - decrease w
        w = w - LR
    elif dw < 0:
        # Slope is negative - increase w
        w = w + LR

# Plot the function and the weights and losses in our epochs
from matplotlib import pyplot as plt

# Create an array of weight values
wRange = np.linspace(-1, 7)

# Use the function to get the corresponding loss values
lRange = [lossfunc(i) for i in wRange]

# Plot the function line
plt.xlabel('Weight')
plt.ylabel('Loss')
plt.grid()
plt.plot(wRange,lRange, color='grey', ls="--")

# Plot the weights and losses we recorded
plt.scatter(weights,losses, c='red')
e = 0
while e < len(weights):
    plt.annotate('E' + str(e+1),(weights[e], losses[e]))
    e += 1

plt.show()

'''
Exploring the Iris Dataset

Before we start using PyTorch to create a model, let's examine the iris dataset. 
Since this is a commonly used sample dataset, it is built-in to the scikit-learn machine learning library, so we'll get it from there. 
As with any supervised learning problem, we'll then split the dataset into a set of records with which to train the model, 
and a smaller set with which to validate the trained model.
'''
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

   
# Split data 70%-30% into training set and test set
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.40, random_state=0)

print ('Training Set: %d, Test Set: %d \n' % (len(x_train), len(x_test)))
print("Sample of features and labels:")
print('(features: ',iris.feature_names, ')')

# Take a look at the first 25 training features and corresponding labels
for n in range(0,24):
    print(x_train[n], y_train[n], '(' + iris.target_names[y_train[n]] + ')')


'''
Importing the PyTorch Libraries

Since we plan to use PyTorch to create our iris classifier, we'll need to install and import the PyTorch libraries we intend to use. 
The specific installation of of PyTorch depends on your operating system and whether your computer has graphics processing units (GPUs) 
that can be used for high-performance processing via cuda. You can find detailed instructions at https://pytorch.org/get-started/locally/.
'''
#pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
    
import torch
import torch.nn as nn
import torch.utils.data as utils
import torch.utils.data as td

print("Libraries imported - ready to use PyTorch", torch.__version__)

'''

Prepare the Data for PyTorch

PyTorch makes use of data loaders to load training and validation data in batches. We've already loaded the data into NumPy arrays, 
but we need to wrap those in PyTorch datasets (in which the data is converted to PyTorch tensor objects) and create loaders to read 
batches from those datasets.
'''
# Create a dataset and loader for the training data and labels
train_x = torch.Tensor(x_train).float()
train_y = torch.Tensor(y_train).long()
train_ds = utils.TensorDataset(train_x,train_y)
train_loader = td.DataLoader(train_ds, batch_size=10,
    shuffle=False, num_workers=1)

# Create a dataset and loader for the test data and labels
test_x = torch.Tensor(x_test).float()
test_y = torch.Tensor(y_test).long()
test_ds = utils.TensorDataset(test_x,test_y)
test_loader = td.DataLoader(test_ds, batch_size=10,
    shuffle=False, num_workers=1)

print("Loaders ready")

'''

Define a Neural Network

Now we're ready to define our neural network. In this case, we'll create a network that consists of 3 fully-connected layers:

    An input layer that receives four input values (the iris features) and applies a ReLU activation function.
    A hidden layer that receives ten inputs and applies a ReLU activation function.
    An output layer that uses a SoftMax activation function to generate three outputs (which represent the probabilities for the three iris species)

'''
# Number of hidden layer nodes
hl = 10

# Define the neural network
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, hl)
        self.fc2 = nn.Linear(hl, hl)
        self.fc3 = nn.Linear(hl, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x),dim=1)
        return x

# Create a model instance from the network
model = IrisNet()
print(model)


'''
Train the Model

To train the model, we need to repeatedly feed the training values forward through the network, use a loss function to calculate the loss, 
use an optimizer to backpropagate the weight and bias value adjustments, and validate the model using the test data we withheld.

To do this, we'll create a function to train and optimize the model, and function to test the model. Then we'll call 
these functions iteratively over 100 epochs, logging the loss and accuracy statistics for each epoch.
'''
def train(model, data_loader, optimizer):
    # Set the model to training mode
    model.train()
    train_loss = 0
    
    for batch, tensor in enumerate(data_loader):
        data, target = tensor
        #feedforward
        optimizer.zero_grad()
        out = model(data)
        loss = loss_criteria(out, target)
        train_loss += loss.item()

        # backpropagate
        loss.backward()
        optimizer.step()

    #Return loss
    avg_loss = train_loss / len(data_loader.dataset)
    return avg_loss
           
            
def test(model, data_loader):
    # Switch the model to evaluation mode (so we don't backpropagate)
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch, tensor in enumerate(data_loader):
            data, target = tensor
            # Get the predictions
            out = model(data)

            # calculate the loss
            test_loss += loss_criteria(out, target).item()

            # Calculate the accuracy
            _, predicted = torch.max(out.data, 1)
            correct += torch.sum(target==predicted).item()
            
    # return validation loss and prediction accuracy for the epoch
    avg_accuracy = correct / len(data_loader.dataset)
    avg_loss = test_loss / len(data_loader.dataset)
    return avg_loss, avg_accuracy
       


# Specify the loss criteria (CrossEntropyLoss for multi-class classification)
loss_criteria = nn.CrossEntropyLoss()

# Specify the optimizer (we'll use a Stochastic Gradient Descent optimizer)
learning_rate = 0.01
learning_momentum = 0.9
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=learning_momentum)

# We'll track metrics for each epoch in these arrays
epoch_nums = []
training_loss = []
validation_loss = []

# Train over 100 epochs
epochs = 100
for epoch in range(1, epochs + 1):
    
    # Feed the training data into the model to optimize the weights
    train_loss = train(model, train_loader, optimizer)
    
    # Feed the test data into the model to check its performance
    test_loss, accuracy = test(model, test_loader)
    
    # Log the metrcs for this epoch
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)
    
    # Print stats for every 10th epoch so we can see training progress
    if (epoch) % 10 == 0:
        print('Epoch {:d}: Training loss= {:.4f}, Validation loss= {:.4f}, Accuracy={:.4%}'.format(epoch, train_loss, test_loss, accuracy))


'''

Review Training and Validation Loss

After training is complete, we can examine the loss metrics we recorded while training and validating the model. We're really looking for two things:

    The loss should reduce with each epoch, showing that the model is learning the right weights and biases to predict the correct labels.
    The training loss and validations loss should follow a similar trend, showing that the model is not overfitting to the training data.

Let's plot the loss metrics and see:
'''
from matplotlib import pyplot as plt

plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()


'''
View the Learned Weights and Biases

The trained model consists of the final weights and biases that were determined by the optimizer during training. Based on our network model we should expect the following values for each layer:

    Layer 1: There are four input values going to ten output nodes, so there should be 10 x 4 weights and 10 bias values.
    Layer 2: There are ten input values going to ten output nodes, so there should be 10 x 10 weights and 10 bias values.
    Layer 3: There are ten input values going to three output nodes, so there should be 3 x 10 weights and 3 bias values.

'''
for param_tensor in model.state_dict():
    print(param_tensor, "\n", model.state_dict()[param_tensor].numpy())


'''

Evaluate Model Performance

So, is the model any good? The raw accuracy reported from the validation data would seem to indicate that it predicts pretty well; 
but it's typically useful to dig a little deeper and compare the predictions for each possible class. A common way to visualize 
the performace of a classification model is to create a confusion matrix that shows a crosstab of correct and incorrect predictions for each class.
'''
#Pytorch doesn't have a built-in confusion matrix metric, so we'll use SciKit-Learn
from sklearn.metrics import confusion_matrix

# Set the model to evaluate mode
model.eval()

# Get predictions for the test data
x = torch.Tensor(x_test).float()
_, predicted = torch.max(model(x).data, 1)

# Plot the confusion matrix
cm = confusion_matrix(y_test, predicted.numpy())
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(iris.target_names))
plt.xticks(tick_marks, iris.target_names, rotation=45)
plt.yticks(tick_marks, iris.target_names)
plt.xlabel("Predicted Species")
plt.ylabel("True Species")
plt.show()


'''


The confusion matrix should show a strong diagonal line indicating that there are more correct than incorrect predictions for each class.
Using the Model with New Data

Now that we have a model we believe is reasonably accurate, we can use it to predict the species of new iris observations:
'''
x_new = [[6.6,3.2,5.8,2.4]]
print ('New sample: {}'.format(x_new[0]))

model.eval()

# Get a prediction for the new data sample
x = torch.Tensor(x_new).float()
_, predicted = torch.max(model(x).data, 1)

print('Prediction:',iris.target_names[predicted.item()])