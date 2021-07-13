[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"

# Dog Breed Classifier Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
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

## Project Motivation <a name="motivation"></a>
Welcome to the Convolutional Neural Networks (CNN) project in the AI Nanodegree! In this project, you will learn how to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, your algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

![Sample Output][image1]

Along with exploring state-of-the-art CNN models for classification, you will make important design decisions about the user experience for your app.  Our goal is that by completing this lab, you understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline.  Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.  Your imperfect solution will nonetheless create a fun user experience!

At the end of the project was made a web page that given an image it predicts with the trained model what is the most likely dog breed of the image:

### Application web url: [https://dog-breed-dsnano-final.herokuapp.com/](https://dog-breed-dsnano-final.herokuapp.com/)

This application could be improved by adding more filters to the images and just receive images with dogs or humans like the example on the notebook.
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