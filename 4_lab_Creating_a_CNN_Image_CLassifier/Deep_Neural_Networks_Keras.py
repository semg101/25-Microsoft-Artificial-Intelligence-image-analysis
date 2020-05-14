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

Before we start using Keras to create a model, let's examine the iris dataset. Since this is a commonly used sample dataset, 
it is built-in to the scikit-learn machine learning library, so we'll get it from there. As with any supervised learning problem, 
we'll then split the dataset into a set of records with which to train the model, and a smaller set with which to validate the trained model.
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
The features are the measurements for each iris observation, and the label is a numeric value that indicates the species of iris that the observation 
represents (versicolor, virginica, or setosa).
Import Keras Libraries

Since we plan to use Keras to create our iris classifier, we'll need to install and import the Keras libraries we intend to use. 
Keras is already installed in Azure Notebooks, but we'll ensure its updated to the latest version. If you're using your own Jupyter 
instance you may need to install Keras and one of the backend frameworks on which it works (Theanos, TensorFlow, or CNTK). 
You can find detailed instructions at https://keras.io/.

'''
#pip install --upgrade keras

import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.utils import np_utils
from keras import optimizers
from keras import backend as K

print("Libraries imported.")
print('Keras version:',keras.__version__)

'''

Prepare the Data for Keras

We've already loaded our data and split it into training and validation datasets. However, we need to do some further data preparation 
so that our data will work correctly with Keras. Specifically, we need to set the data type of our labels to 32-bit floating point numbers, 
and specify that the labels represent categorical classes rather than numeric values.
'''
# Set data types for float features
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Set data types for categorical labels
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print("Data ready")


'''

Define a Neural Network

Now we're ready to define our neural network. In this case, we'll create a network that consists of 3 fully-connected layers:

    An input layer that receives four input values (the iris features) and applies a ReLU activation function to produce ten outputs.
    A hidden layer that receives ten inputs and applies a ReLU activation function to produce another ten outputs.
    An output layer that uses a SoftMax activation function to generate three outputs (which represent the probabilities for the three iris species)

'''
# Define a classifier network
hl = 10 # Number of hidden layer nodes

model = Sequential()
model.add(Dense(hl, input_dim=4, activation='relu'))
model.add(Dense(hl, input_dim=hl, activation='relu'))
model.add(Dense(3, activation='softmax'))

print(model.summary())


'''

Train the Model

To train the model, we need to repeatedly feed the training values forward through the network, use a loss function to calculate the loss, 
use an optimizer to backpropagate the weight and bias value adjustments, and validate the model using the test data we withheld.

To do this, we'll apply a Stochastic Gradient Descent optimizer to a categorical cross-entropy loss function iteratively over 100 epochs.
'''
# hyper-parameters for optimizer
learning_rate = 0.01
learning_momentum = 0.9
sgd = optimizers.SGD(lr=learning_rate, momentum = learning_momentum)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# Train the model over 100 epochs using 10-observation batches and using the test holdout dataset for validation
num_epochs = 100
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=10, validation_data=(x_test, y_test))


'''

Review Training and Validation Loss

After training is complete, we can examine the loss metrics we recorded while training and validating the model. We're really looking for two things:

    The loss should reduce with each epoch, showing that the model is learning the right weights and biases to predict the correct labels.
    The training loss and validations loss should follow a similar trend, showing that the model is not overfitting to the training data.

Let's plot the loss metrics and see:
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

View the Learned Weights and Biases

The trained model consists of the final weights and biases that were determined by the optimizer during training. Based on our network model we should expect the following values for each layer:

    Layer 1: There are four input values going to ten output nodes, so there should be 10 x 4 weights and 10 bias values.
    Layer 2: There are ten input values going to ten output nodes, so there should be 10 x 10 weights and 10 bias values.
    Layer 3: There are ten input values going to three output nodes, so there should be 10 x 3 weights and 3 bias values.

'''
for layer in model.layers:
    weights = layer.get_weights()[0]
    biases = layer.get_weights()[1]
    print('------------\nWeights:\n',weights,'\nBiases:\n', biases)


'''

Evaluate Model Performance

So, is the model any good? The raw accuracy reported from the validation data would seem to indicate that it predicts pretty well; 
but it's typically useful to dig a little deeper and compare the predictions for each possible class. A common way to visualize 
the performace of a classification model is to create a confusion matrix that shows a crosstab of correct and incorrect predictions for each class.

'''
#Keras doesn't have a built-in confusion matrix metric, so we'll use SciKit-Learn
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class_probabilities = model.predict(x_test)
predictions = np.argmax(class_probabilities, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Plot the confusion matrix
cm = confusion_matrix(true_labels, predictions)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(iris.target_names))
plt.xticks(tick_marks, iris.target_names, rotation=85)
plt.yticks(tick_marks, iris.target_names)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()


'''


The confusion matrix should show a strong diagonal line indicating that there are more correct than incorrect predictions for each class.
Using the Model with New Data

Now that we have a model we believe is reasonably accurate, we can use it to predict the species of new iris observations:
'''
x_new = np.array([[6.6,3.2,5.8,2.4]])
print ('New sample: {}'.format(x_new[0]))

class_probabilities = model.predict(x_new)
predictions = np.argmax(class_probabilities, axis=1)

print(iris.target_names[predictions][0])

