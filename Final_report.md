# Detecting AI-generated content (AIGC)

**Author:** LAM Yan Yi, Elaine (57150480), HUNG Kai Hin (57137090), WONG Hoi Fai (57151396)

**Date:** 6-Dec-2024

## Abstract

## Table of Contents

1. [Feature Extraction](#feature-extraction) <br>
    1.1 [Error Level Analysis](#error-level-analysis-ela) <br>
    1.2 [Morphological Filter](#morphological-filter-mf) <br>
    1.3 [Local Binary Pattern](#local-binary-pattern) <br>
2. [Data Loader](#data-loader) <br>
    2.1 [Data Loading](#data-loading) <br>
    2.2 [Data Sampling](#data-sampling) <br>
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
### SE-ResNeXt Model
SE-ResNeXt is a neural network combining ResNext, Squeeze-and-Excitation blocks and the principle of cardinality for performing dynamic channel-wise feature recalibration. ResNeXt repeats building blocks that aggregate many transformations. 

To achieve the goal of enhancing the representational power of the network, a parameter "cardinality" is implemented to control the paths through the network. Cardinality refers to the amount of parallel pathways within any given building block in the network. Through increasing the cardinality, it greatly improves the model's ability to capture a plethora of features without significantly increasing the number of parameters, allowing the network to better grasp the convoluted connections between features. It is achieved by enabling different paths that focus on various aspects of the input data. The increased cardinality effectively utilises the computation capacity to capture a broader range of features to ultimately enhance its representational power, leading to improved performance in sophisticated relationship mapping between features and labels. The implementation of the cardinality principle distinctly separates and distinguishes SE-ResNeXt from traditional ResNet networks.

ResNeXt is a neural network architecture extended from ResNet that introduces a new block structure. It puts a strong emphasis on parallelising the model to improve the scalability of the model. The improved scalability allows for more efficient use of computational power, increasing the performance of the model. The ResNeXt network consists of many blocks, where each block is comprised of a sequence of convolutional layers and ends after batch normalisation and the ReLU activation function is applied. For each of the blocks, the input data is separated into multiple channels based on the aforementioned cardinality parameter. The outputs are condensed before passing to the next block.

For Squeeze-and-Excitation block, it is a neural network component designed to provide a more comprehensive view of the features, thereby increasing the representational power of the network. It achieves a comprehensive view through adaptive recalibration of the feature maps. The "Squeeze" operation typically uses global average pooling to greatly reduce the spatial dimensionality of the feature maps to a 1x1 matrix. On the other hand, the "Excitation" operation learns a channel-wise/feature-wise weighting that encapsulates the significance of each of the given feature channels. The channel-wise weights are learnt through a smaller neural network. The Squeeze-and-Excitation block incorporates both the squeeze and excitation operations. The input feature maps are passed through the squeeze operation to obtain a channel-wise descriptor of the features. Then, it is passed through the excitation mechanism to learn the forenamed channel-wise weightings to scale the feature maps accordingly.

SE-ResNeXt combines the three factors mentioned above to achieve higher accuracy in image classification tasks. The SE blocks improve feature selection while allowing us to train our model with separated channels for various feature extraction methods.