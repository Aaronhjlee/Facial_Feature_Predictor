<!-- ![](images/doodle_predictor_banner.png) -->

# Facial_Feature_Predictor
**Using headshots to predict gender and attractiveness**
<br>Aaron Lee
<br>
[Linkedin](http://www.linkedin.com/in/aaronhjlee)  |  [Github](https://github.com/aaronhjlee)   |   aaronhjlee1@gmail.com

## Table of Contents

* [Motivation](#motivation)
  * [Personal](#personal)
  * [Question](#question)
* [Strategy](#strategy)
* [Exploratory Data Analysis](#exploratory-data-analysis)
* [Machine Learning Modeling](#machine-learning-models)
* [Conclusion](#conclusion)
* [Future Analysis](#future-analysis)
* [Tools Used](#tools-used)
* [References](#references)
* [Contact](#contact-information)

## Motivation

Photography has been a hobby of mine for the past 7 years and inparticular, portrait photography. There are a lot of aspects to consider when taking a good picture. When I was starting out, one of my first paid gigs was to do headshots of lawyers. I would tell them to angle their shoulders one way, tilt their chin another, rest their eyes before the moment, etc. I know for a fact, my first time was not my best work, and very far from it. Like how I would not blame my camera for taking a bad photo, I would never blame my subject for making the photo turn out 'undesireable'. I knew there was a way to make **anyone** look good in front of the camera, it was all about perspective. 

### Personal

<<<<<<< HEAD
So why did I choose to analyze headshots from a dataset of celebrities? It was an easy way to collect data and to start building a model on people, who by in large, deemed attractive. By having a computer take inventory on the myriad of features that come with an individual's face, there would be an unbiased metric of parsing which facial features mattered in determining an individual's picture was attractive or not. 
=======
So why did I choose to analyze headshots from a dataset of celebrities? It was an easy way to collect data and to start building a model on people, who by in large, deemed attractive. By having a computer take inventory on the myriad of features that come with an individual's face, there would be an unbiased metric of parsing which facial features mattered in determining an individual's picture was attractive or not.
>>>>>>> individual

### Question

What facial features determine an attractive headshot?

## Strategy

#### 1. Load and Clean Data
#### 2. Exploratory Data Analysis
* Transform with Principal Component Analysis
* Find Ideal Variance Percentage / Number of Components
#### 3. Convolutional Neural Network
* Determine male / female
* Predict probability of attractiveness based on facial features
#### 4. Feature Importance 
* Extract the filters to determine which features are important

## Exploratory Data Analysis

### Data Overview
  1. Dataset: 200000+ faces with 40 attributes from [The Chinese University of Hong Kong](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  2. Some attributes include: male, attractive, oval face, big nose, 5 O'clock shadow, big lips, rosy cheeks, heavy makeup, etc. 
  3. 183 unique countries out of 196 countries (UN considers 241 countries-fun fact!)
  4. Each drawing contained information about the number of strokes, location, timestamp, recognition(binary), and label
  5. Sample 5000 images randomly from each facial feature category for a grand total of 20,000 images in our available dataset
  6. Google's [Neural Network](https://adventuresinmachinelearning.com/python-tensorflow-tutorial/) was able to achieve a recognition rate of over *91%* across 345 different categories of doodles
  7. Drawing Examples
Below are drawings that Google's NN recognized and followed by drawings that were not. 

## Future Analysis

## Tools Used

* [Python](https://www.python.org/)
* [Pandas](https://pandas.pydata.org/)
* [Numpy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [Scikit-Learn](https://scikit-learn.org/stable/)
* [Tf.Keras](https://www.tensorflow.org/guide/keras)

## References

* http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

## Contact Information
Aaron Lee is a Data Scientist with a background in education and portrait/landscape photography. His interests other than data science include music, photography, backpacking, basketball, and running. 

* Linkedin: [in/aaronhjlee](https://www.linkedin.com/in/aaronhjlee/)
* Github: [/aaronhjlee](https://github.com/Aaronhjlee)
* Email: [aaronhjlee1@gmail.com](aaronhjlee1@gmail.com)

Copyright © 2019 Aaron Lee