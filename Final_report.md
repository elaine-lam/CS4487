# Detecting AI-generated content (AIGC)

**Author:** LAM Yan Yi, Elaine (57150480), HUNG Kai Hin (57137090), WONG Hoi Fai (57151396)

**Date:** 6-Dec-2024

## Abstract

## Table of Contents

1. [Feature Extraction](#feature-extraction) <br>
    1.1 [Error Level Analysis](#error-level-analysis-ela) <br>
    1.2 [Morphological Filter](#morphological-filter-mf) <br>
    1.3 [Local Binary Pattern](#local-binary-pattern) <br>
2. [Data Loader](#data-loader)
3. [Model](#model)

## Feature Extraction
Feature extraction is a process to extract relevant information from raw data. Instead of putting every possible data into the model, a few selected features are used as input for the model. Unrelevant data is disregarded to reduce the drain on computational power and reduce the chances of confusing the network. The simplified data representation makes the training more efficient and quicker to complete, reducing training time.

However, choosing the suitable features to extract is no easy feat. Extracting unrelated features puts an unnecessary drain on the system and induces the possibility of confusing the training network. However, not extracting crucial features leaves the network pondering for more useful data and, therefore, unable to map useful connections between the inputs and the labels.

### Error Level Analysis (ELA)
Error level analysis is a common digital forensic technique to recognise images that have been tampered with or digitally altered. Error level analysis detects irregular distributions of quantisation noise to help identify possible regions of the image with a high concentration of inconsistencies in error level across the image.

Leveraging the ability of the error level analysis technique, we implemented it into our project as one of our feature extraction methods. It provided our system a way to highlight regions of interest that displayed a significant difference in in error levels. Due to the fact that AI-generated images often do not include natural imperfections found in real photos, it would serve a great purpose in providing information for the network to determine whether it is AI-generated or photorealistic images. 


### Morphological Filter (MF)
Morphological filters are a class of image processing techniques used to analyse and process the shapes and structures within an image. There are two types of basic morphological filters as well as two advanced morphological filters that utilise the two basic ones in conjunction.

The first basic morphological filter is erosion. A morphological erosion filter is a robust technique for shrinking the boundaries of objects within an image. It decreases the amount of bright regions while increasing the amount of dark regions. By decreasing the overall amount of bright regions, it is able to erode away the brighter regions at the boundaries of the objects, effectively pushing the boundaries inward towards the centre of the object space. This process is done iteratively, resulting in an image with a greatly reduced amount of bright regions. This reduction in bright pixels is achieved through the iterative application of a minimising kernel. After applying a mask or a kernel to the image, the output pixel is calculated to be the minimum value of all the values within the masked area. The resulting output image has a strong tendency to separate overlapping objects as each boundary of the objects is shrunk inwards respectively.

The other form of basic morphological filter is dilation. A morphological dilation filter is highly capable in expanding the region of interest within an image. Reflecting a strong contrast from that of a morphological erosion filter, it increases the amount of bright regions while decreasing the amount of dark regions. By increasing the overall amount of bright areas, it is able to fill in small gaps and holes within objects in the image. On a large enough scale, it can even connect disjointed parts of the same object by enlarging and thickening the object's visual boundaries. The boundaries are seen as moving outwards, away from the centre of the object. This process is done iteratively, resulting in an image with a greatly reduced amount of dark regions. The increase in bright pixels is achieved through the iterative application of a maximising kernel. After the maximising kernel is applied, the output pixel is calculated to be the maximum value of all the values contained in the kernel area. The resulting output image will have a high likelihood of filling small gaps in the object. The added benefit of a morphological dilation filter is its ability to smooth out rough boundaries.

By combining the two basic types of morphological filters, two advanced types of morphological filters came to fruition: Opening and Closing.

Opening combines erosion and dilation, where the input image first passes through a morphological erosion filter and then passes through a morphological dilation filter. Through the specific ordered combination of erosion and dilation, it is able to break apart narrow gaps between objects. An added benefit is small objects will be covered by the filter, allowing extraction only of the important major objects.

Closing also combines both erosion and dilation. However, it encompasses both in a different order. Unlike opening, it first uses a morphological dilation filter, and then the resulting image is passed through a morphological erosion filter. Through the specific ordered combination of erosion and dilation, it is able to close small breaks.

Through careful consideration, we have decided to use opening as our method of feature extraction for our detection between AI-generated content and photorealistic images.

### Local Binary Pattern
Local binary pattern is a popular texture feature extraction technique used in the realm of machine learning and data analytics. It is able to provide a strong description of the local texture patterns within an image through comparing the central pixel with its neighbouring pixels to represent in a binary pattern. If the neighbouring pixels cross a threshold and are more intense than the centre pixel, a binary "1" is assigned. Through iterative computation of the local binary pattern for each of the pixels on the screen, the resulting output can accurately represent the local textural information stored within the image.

## Data Loader
### Data Loading
Our data loader iteratively goes through the data path, storing all of the image file paths, and their respective labels are assigned based on the folder they reside in. The image file paths are later opened to load the images as tensors. The tensors are converted to NumPy arrays to prepare for feature extraction. Before closing the image and disregarding the NumPy arrays, the arrays are passed through the three different feature extraction methods mentioned above. The three extracted feature channels are stacked and made into torch tensors. 

There is error checking to handle unforeseen cases of faulty image loading. If, for whatever reason, the image is unable to be converted to an image tensor, the system will log a critical logging message indicating what data type is received instead of the required tensor type. This not only explains to the user why the system has crashed but also provides helpful information to developers for an easier debugging process.

We have decided to use the extracted image folder instead of using the zip folder directly due to efficiency concerns. Instead of having to unzip the zip folder for each run of the program, our implementation assumes the zip folder has already been unzipped into a vanilla directory. It allows for a shorter loading time and greatly enhances the computational efficiency of the program, resulting in a shorter training time.

### Data Sampling
Not all of the data loaded is mixed together to be delivered to the network for training and validation. The training dataset is extracted and split into batches, randomising the order of the data with shuffling and then put into the network for training. The shuffling is done randomly and differently for each epoch, so the training data is different for each iteration of the training process. It gives the much-needed variety for the training data, allowing the network to have a more comprehensive view of the data. However, for the validation dataset, it is a drastically different story. For the validation dataset, we purposefully did not add random shuffling after each epoch. This is because the validating data has to be evaluated fairly between each epoch. Therefore, it is paramount that it stays the same to provide a static and unbiased view of the performance of the current epoch.

## Model