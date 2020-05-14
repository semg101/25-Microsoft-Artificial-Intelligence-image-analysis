'''
Processing Images
As far as computers are concerned, images are simply numerical data representations. You can use statistical techniques to manipulate and analyze the numerical properties of images.
Loading and Displaying an Image
Let's start by loading a JPG file and examining its properties. Run the following cell to load and display an image using the matplotlib.image library.
'''

from matplotlib import image as mpimg
from matplotlib import pyplot as plt
import numpy as np

img1 = mpimg.imread('data/graeme.jpg')
plt.imshow(img1)
print(type(img1))

'''
So we can see the file is definitely an image, but note that the data type of the img1 object is actually a multidimensional numpy array.
Let's take a closer look at the shape of this array:
'''
print(img1.shape)


'''
The image is actually composed of three "layers, or channels, for red, green, and blue (RGB) pixel intensities. 
Each layer of the image represents 433 x 650 pixels (the dimensions of the image).
Now let's load and display the same image but this time we'll use another popular Python library for working with images - PIL.
'''
from PIL import Image
import matplotlib.pyplot as plt

img2 = Image.open('data/graeme.jpg')
plt.imshow(img2)
print(type(img2))

'''
This time, the data type is a JpegImageFile - not a numpy array. That's great if we only want to manipulate it using PIL methods; 
but sometimes we'll want to be flexible and process images using multiple libraries; so we need a consistent format.
Fortunately, it's easy to convert a PIL JpegImageFile to a numpy array:
'''
import numpy as np

img2 = np.array(img2)
print(img2.shape)

#You can also convert a numpy array to a PIL image, like this:
img3 = Image.fromarray(img1)
plt.imshow(img3)
print(type(img3))

'''
So fundamentally, the common format for image libraries is a numpy array. 
Using this as the standard format for your image processing workflow, converting to and from other formats as required, 
is the best way to take advantage of the particular strengths in each library while keeping your code consistent.
You can even save a numpy array in an optimized format, should you need to persist images into storage:
'''
import numpy as np

# Save the image
np.save('data/img.npy', img1)

#Load the image
img4 = np.load('data/img.npy')

plt.imshow(img4)

'''
Resizing an Image
One of the most common manipulations of an image is to resize it. This can be particularly important when you're preparing 
multiple images to train a machine learning model, as you'll generally want to ensure that all of your training images have consistent dimensions.
Let's resize our image:
'''
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Load the image array into a PIL Image
orig_img = Image.fromarray(img4)

# Get the image size
o_h, o_w = orig_img.size
print('Original size:', o_h, 'x', o_w)

# We'll resize this so it's 200 x 200 using the thumbnail metho
target_size = (200,200)
new_img = orig_img.resize(target_size)
n_h, n_w = new_img.size
print('New size:', n_h, 'x', n_w)

# Show the original and resized images
# Create a figure
fig = plt.figure(figsize=(12, 12))

# Subplot for original image
a=fig.add_subplot(2,1,1)
imgplot = plt.imshow(orig_img)
a.set_title('Before')

# Subplot for resized image
a=fig.add_subplot(2,1,2)
imgplot = plt.imshow(new_img)
a.set_title('After')

plt.show()

'''
Well, that worked; but notice that the image is not scaled. We resized the rectangular image to have square dimensions, 
and the image is skewed to fill the new size. If we want to resize the image and change its shape without distorting it, 
we'll need to scale the image so that its largest dimension fits our new desired size, and fill out any additional space with some sort of border.
'''
from PIL import Image, ImageOps

# We'll resize ththe image so it's 200 x 200
target_size = (200,200)

# Create a figure
fig = plt.figure(figsize=(12, 12))

# Load the original image
orig_img = Image.fromarray(img1)
orig_height, orig_width = orig_img.size
print('Original size:', orig_height, 'x', orig_width)
# Plot the image
a=fig.add_subplot(3,1,1)
imgplot = plt.imshow(orig_img)
a.set_title('Original')

# Scale the image to the new size using the thumbnail method
scaled_img = orig_img
scaled_img.thumbnail(target_size, Image.ANTIALIAS)
scaled_height, scaled_width = scaled_img.size
print('Scaled size:', scaled_height, 'x', scaled_width)
# Plot the scaled image
a=fig.add_subplot(3,1,2)
imgplot = plt.imshow(scaled_img)
a.set_title('Scaled')

# Create a new white image of the target size to be the background
new_img = Image.new("RGB", target_size, (255, 255, 255))
# paste the scaled image into the center of the white background image
new_img.paste(scaled_img, (int((target_size[0] - scaled_img.size[0]) / 2), int((target_size[1] - scaled_img.size[1]) / 2)))
new_height, new_width = new_img.size
print('New size:', new_height, 'x', new_width)
# Plot the resized image
a=fig.add_subplot(3,1,3)
imgplot = plt.imshow(new_img)
a.set_title('Resized')

plt.show()

'''
Examining Numerical Properties of the Image Array
So we've seen that an image is inherently an array of values. Let's examine that in more detail. 
What type of values are in the array?
'''
print(img1.dtype)

'''
OK, so the array consists of 8-bit integer values. In other words, whole numbers between 0 and 255. 
These represent the possible pixel intensities for the RGB color channels in each pixel of the image.
Let's look at the distribution of pixel values in the image. Ideally, the image should have relatively even distribution of values, 
indicating good contrast and making it easier to extract analytical information.
An easy way to check this is to plot a histogram.
'''
import matplotlib.pyplot as plt

# Plot a histogram - we need to use ravel to "flatten" the 3 dimensions
plt.hist(img1.ravel())
plt.show()

'''
Another useful way to visualize the statistics of an image is as a cumulative distribution function (CDF) plot. 
Which shows the cumulative pixel intensity frequencies from 0 to 255.
'''
import matplotlib.pyplot as plt

plt.hist(img1.ravel(), bins=255, cumulative=True)
plt.show()

'''
The histogram and CDF for our image show pretty uneven distribution - there's a lot of contrast in the image. 
Ideally we should equalize the values in the images we want to analyse to try to make our images more consistent in terms of 
the shapes they contain irrespective of light levels.
Histogram equalization is often used to improve the statistics of images. In simple terms, the histogram equalization algorithm attempts 
to adjust the pixel values in the image to create a more uniform distribution. The code in the cell below uses the exposure.equalize_hist method 
from the skimage package to equalize the image.
> You can ignore the warning displayed by this code
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import exposure
%matplotlib inline

img_eq = exposure.equalize_hist(img1)

# Display using matplotlib

# Create a figure
fig = plt.figure(figsize=(16, 8))

# Subplot for original image
a=fig.add_subplot(1,2,1)
imgplot = plt.imshow(img1)
a.set_title('Before')

# Subplot for processed image
a=fig.add_subplot(1,2,2)
imgplot = plt.imshow(img_eq)
a.set_title('After')

plt.show()


#As with most image operations, there's more than one way to do this. For example, you could also use the PIL.ImgOps.equalize method:
from PIL import Image, ImageOps

# Equalize the image - but we need to convert the numpy array back to the PIL image format
imgPIL_eq = ImageOps.equalize(Image.fromarray(img1))

# Display using matplotlib

# Create a figure
fig = plt.figure(figsize=(16, 8))

# Subplot for original image
a=fig.add_subplot(1,2,1)
imgplot = plt.imshow(img1)
a.set_title('Before')

# Subplot for processed image
a=fig.add_subplot(1,2,2)
imgplot = plt.imshow(imgPIL_eq)
a.set_title('After')

plt.show()


#Now let's see what that's done to the histogram and CDF plots:
# Display histograms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

img_eq = np.array(imgPIL_eq)

# Create a figure
fig = plt.figure(figsize=(16, 8))

# Subplot for original image
a=fig.add_subplot(1,2,1)
imgplot = plt.hist(img_eq.ravel())
a.set_title('Histogram')

# Subplot for processed image
a=fig.add_subplot(1,2,2)
imgplot = plt.hist(img_eq.ravel(), bins=255, cumulative=True)
a.set_title('CDF')

plt.show()

'''
The pixel intensities are more evenly distributed in the equalized image. 
In particular, the cumulative density shows a straight diagonal cumulation; which is a good sign that the pixel intensity values have been equalized.
'''

#Denoising with Filters
'''
Often images need to be cleaned up for analysis. One way to do this is to apply a filter.

A filter is a small patch (or kernel) that is applied repeatedly (the correct term is convolutionally) across the image. 
For example, lets imaging this matrix represents a small, 1-dimensional image:
⎡⎣⎢⎢⎢⎢⎢⎢⎢⎢255255255255255255255255125125255255255125125125125255255125125125125255255255125125255255255255255255255255⎤⎦⎥⎥⎥⎥⎥⎥⎥⎥

We could process this image wih a 3x3 filter. The first application of the filter would be to the red pixels below:
⎡⎣⎢⎢⎢⎢⎢⎢⎢⎢255255255255255255255255125125255255255125125125125255255125125125125255255255125125255255255255255255255255⎤⎦⎥⎥⎥⎥⎥⎥⎥⎥

Now, let's suppose that this filter affects the image by recalculating the center pixel value as the mean of the pixel values in 
the kernel patch (this kind of filter is known as a mean filter and has the effect of blurring the image). Here's what that does 
to the first patch of the image:
⎡⎣⎢⎢⎢⎢⎢⎢⎢⎢255255255255255255255212125125255255255125125125125255255125125125125255255255125125255255255255255255255255⎤⎦⎥⎥⎥⎥⎥⎥⎥⎥

Then we move the filter along and apply it to the next patch:
⎡⎣⎢⎢⎢⎢⎢⎢⎢⎢255255255255255255255212125125255255255128125125125255255125125125125255255255125125255255255255255255255255⎤⎦⎥⎥⎥⎥⎥⎥⎥⎥

This process is repeated until the filter has been convolved across the entire image. Filtersv like this can be used to sharpen 
or blur images, and are often used to reduce the affect of scatter or noise in an image.
'''

#Add Some Random Noise

#To see how this works, let's first add some random noise to our image - such as you might see in a photograph taken in low light or at a low resolution.
import skimage

img_n = skimage.util.random_noise(img_eq)
plt.imshow(img_n)

#Using a Gaussian Filter
#A Gaussian filter is a slightly more complex version of a mean filter, with a similar blurring effect.
from scipy.ndimage.filters import gaussian_filter as gauss

img_gauss = gauss(img_n, sigma=1)   
plt.imshow(img_gauss)

#Using a Median Filter
'''
The Gaussian filter results in a blurred image, which may actually be better for feature extraction as it makes it easier 
to find contrasting areas. If it's too blurred, we could try a median filter, which as the name suggests applies the median value 
to the pixel in the center of the filter kernel.
'''
from scipy.ndimage.filters import median_filter as med

img_med = med(img_n, size=2)
plt.imshow(img_med)

'''
Extract Features

Now that we've done some initial processing of the image to improve its statistics for analysis, we can start to extract features from it.
Sobel Edge Detection

As a first step in extracting features, you will apply the Sobel edge detection algorithm. This finds regions of the image with large gradient values in multiple directions. Regions with high omnidirectional gradient are likely to be edges or transitions in the pixel values. This works by performing two convolutional passes in which the pixel values are multiplied by the following filter kernels:
Vertical⎡⎣⎢−1−2−1000121⎤⎦⎥
Horizontal⎡⎣⎢−101−202−101⎤⎦⎥

The code in the cell below applies the Sobel algorithm to the median filtered image, using these steps:

    Convert the color image to grayscale for the gradient calculation since it is two dimensional.
    Compute the edge gradients in the x and y (horizontal and vertical) directions using the two kernels.
    Compute the magnitude of the gradient vectors identified by the filters.
    Normalize the gradient values.
'''

def edge_sobel(image):
    from scipy import ndimage
    import skimage.color as sc
    import numpy as np
    image = sc.rgb2gray(image) # Convert color image to gray scale
    dx = ndimage.sobel(image, 1)  # horizontal derivative
    dy = ndimage.sobel(image, 0)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag *= 255.0 / np.amax(mag)  # normalize (Q&D)
    mag = mag.astype(np.uint8)
    return mag

img_edge = edge_sobel(img_med)
plt.imshow(img_edge, cmap="gray")


#Now let's try with the more blurred gaussian filtered image.
img_edge = edge_sobel(img_gauss)
plt.imshow(img_edge, cmap="gray")

'''


Note that the lines are more pronounced. Although a gaussian filter makes the image blurred to human eyes, this blurring can actually 
help accentuate contrasting features for computer processing.
Harris Corner Detection

Another example of a feature extraction algorithm is corner detection. In simple terms, the Harris corner detection algorithm 
applies a filter that locates regions of the image with large changes in pixel values in all directions. These regions are said to be corners. 
The Harris corner detector is paired with the corner_peaks method. This operator filters the output of the Harris algorithm, over a patch of 
the image defined by the span of the filters, for the most likely corners.
'''
# Function to apply the Harris corner-detection algorithm to an image
def corner_harr(im, min_distance = 20):
    import skimage.color as sc
    from skimage.feature import corner_harris, corner_peaks
    
    im = sc.rgb2gray(im) # Convert color image to gray scale
    mag = corner_harris(im)
    return corner_peaks(mag, min_distance = min_distance)

# Find the corners in the median filtered image with a minimum distance of 20 pixels
harris = corner_harr(img_med, 20)

print (harris)

# Function to plot the image with the harris corners marked on it
def plot_harris(im, harris, markersize = 20, color = 'red'):
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure(figsize=(6, 6))
    fig.clf()
    ax = fig.gca()    
    ax.imshow(np.array(im).astype(float), cmap="gray")
    ax.plot(harris[:, 1], harris[:, 0], 'r+', color = color, markersize=markersize)
    return 'Done'  

plot_harris(img_med, harris)

