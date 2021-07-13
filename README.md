[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/Labrador_retriever_06449.jpg "Sample Input"
[image3]: ./images/Labrador_retriever_06455.jpg "Sample Input"
[image4]: ./images/Brittany_02625.jpg "Sample Input"
[image5]: ./images/sample_cnn.png "Sample Model"
[image6]: ./images/images_by_breed.png "Sample Images"



# Dog Breed Classifier Project

### Table of Contents

1. [Installation](#installation)
2. [Project Definition](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation<a name="installation"></a>
1. Clone the repository.
2. Create a virtual environment environment with python 3.8.11.
3. Install the requirements
```
pip install -r requirements.txt
```
4. Run the application:
```
python app.py
```
### Datasets
1. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/data/dog_images`. 

2. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/data/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

3. Download the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

## Project Definition <a name="motivation"></a>

### Project Overview
On this project id build a pipeline that can be used within a web or mobile app to processs real-wold, user-supplied images.
I used a dataset with images of different dogs and human to train the model, the datasets are described in the previous section.

### Problem Statement
Given an image of a dog the algorithm will identify an estimate of the canine's breed. If supplied and image of a human, the code will
identify the resembling dog breed.

![Sample Output][image1]

To solve this problem I'm going to explore different techniques with Convolutional Neural Networks (CNN) and strategies like transfer learning to improve the performance of the generated models.
 
### Metrics
To validate the performance of the different models that I'm going to try, I used the accuracy of the model, this is because this is one of the most common used metric. This is applied validating the expected results against the test and validation dataset.
As the dataset seems to be a little imbalanced for future work we could try other of the described metrics on [this article](https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/)

## Analysis
### Data Exploration
The dataset used for this model is made up of images of dogs and humans that are used to traing the model.
This dataset is labeled and contains the following information:
1.  **Dogs Dataset:**
- There are 133 total dog categories.
- There are 8351 total dog images.
- There are 6680 training dog images.
- There are 835 validation dog images.
- There are 836 test dog images.

2. **Human Dataset**
- There are 13233 total human images.

There different images on the training dataset are a little imbalanced as they fluctuate to between 30 and 70 images per label.

The dog breed with more images in the train dataset is: 005.Alaskan_malamute with 77 images and the dog breed with fewer images in the train dataset is: 132.Xoloitzcuintli with 26 images

### Data Visualization
Some of them looks like this:
![Sample Input][image2]
![Sample Input][image3]
![Sample Input][image4]

I created this graph to show the fluctuation between the number of images on the training dataset by dog breed:
![Sample Images][image6]

## Methodology
### Data Preprocessing
For the human face detector we apply a filter to convert the images to gray scale before sending them to the model.
For the dogs' dataset we scale them to 224×224 pixels, and we convert the images to tensors tha have the form of 4D arrays with shape:
(nb_samples,rows,columns,channels)

This process was made because the models needs this structure to work.

### Implementation and Refinement
I explored three different techniques with Convolutional Neural Networks (CNN). The first one was creating a CNN to Classify Dog Breeds (from Scratch) using this model:
![Sample Model][image5]
I tried to change the number of layers and the optimizer but the results doesn't change a lot, it rounds about 3 or 5 percent of accuracy. So I tried to increase the number of epochs and it increases the accuracy to 7%.

The second approach was to use transfer learning to create the CNN using a pre train model, the first model that I used is the VGG-16 and then I only added a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.
This model gave an accuracy of 40%.

The last approach was also with transfer learning but using the pre trained model Resnet50 and also adding a global average pooling layer, and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.
This model gave an accuracy of 79%. 

You could find more information on [this notebook](https://github.com/carogomezt/datascience_final_project/blob/main/dog_app.ipynb).

Finally, after testing different models we took the last one and use it to create a web application with Flask, it receives a user image and returns the dog's breed that is the most similar.

## Results
### Model Evaluation and Validation
For the selected model, the number of epochs that we used to train the model was 20 and as the optimizer we used the _rmsprop_, as it gave a good accuracy I didn't try different options, but if we want to explore more the model we could try other optimizers like _adam_ and also increase the number of epochs to let the model generalize better, but we need to be careful about this approach because it could guide us to an overfitting if we left the model training a lot.
### Justification
The model that gave the best accuracy was the one that used the Resnet50 as an initial step, this is because this is a robust model that was trained in millions of images and generalizes well. 
You could find more information about the Resnet50 model [here](https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33).
The web application was created using the model with the highest accuracy and deployed on Heroku.

## Conclusion
The generated model has an accuracy of 79% and testing it with different scenarios we could see that it generalizes well in difficult escenarios like a person with backgrounds or multiple dogs in the same image. Also it could recognize a cat as a different element from a human or a dog. We could improve the accuracy of the model by testing it with more images to improve the generalization of the model. It would be possible with augmentation of the data (rotating or scaling the images) or adding new images to the dataset. Another option could be to test different transfer learning models like the ones described in the notebook and pick up the one who gives a better performance. And finally we could add more layers to the CNN after using a trained model that could be more robust and improve the accuracy.

At the end of the project was made a web page that given an image it predicts with the trained model what is the most likely dog breed of the image. We could also improve this application by adding the same final step of the notebook, recognizing human faces and dog faces and giving custom messages to the user.

### Application web url: [https://dog-breed-dsnano-final.herokuapp.com/](https://dog-breed-dsnano-final.herokuapp.com/)

## File Descriptions<a name="files"></a>

1. **data**: Folder with .csv files with dogs breeds.
2. **haarcascades**: Folder with the file of pre-train Haar feature-based cascade classifiers for face detection.
3. **images**: Folder with test images and images that are shown in the jupyter notebook.
4. **saved_models**: Folder with the model Resnet50.
5. **static**: Folder in which the uploaded files will be saved.
6. **templates**: Folder with the html templates for the web page.
7. **app.py**: File with the code to run the web page.
8. **dog_app.html**: File with the analysis and model implementation.
9. **dog_app.ipynb**: File with the analysis and model implementation.
10. **extract_bottleneck_features**: File with functions to extract bottleneck features for the model.ç
11. **models.py**: File with data model for the web page.
12. **predictor.py**: File with model for the prediction.
13. **Procfile**: File with instructions to deploy de app.
14. **README.md**: File with repository information.
15. **requirements.txt**: File with requirements of the project.
16. **runtime.txt**: File with requirements of the project.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The Datasets used on this project are being provided by Udacity.