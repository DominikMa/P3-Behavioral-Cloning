# **Behavioral Cloning**

This repository contains my solution for the project "Behavioral Cloning" of the Udacity Self-Driving Car Engineer Nanodegree Program. The python code to generate the model could be found in [model.py](model.py).
<!-- TODO add ref to udacity git https://github.com/udacity/CarND-Behavioral-Cloning-P3 -->

The following part of the README contains a writeup which describes how the behavioral cloning is achieved.

---

## Writeup

### Goals

In the project the goal is to build and train a neuronal net with Keras which should be able to drive through two tracks. Therefor the model receives a picture of the front view of the car and predicts a steering angle.

The goal of this project includes:
* Analyze the given training data
* Generate new training data with the simulator
* Design, train and test a neuronal net
* Analyze the predictions
* Run the model on the simulator and pass at least track one


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


<!-- ## Rubric Points -->
<!-- ### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.   -->

### Included Files

This repository contains all required files to generate, train and test the model and can be used to run the simulator in autonomous mode. It does not contain the training data.

The required files are:
* [model.py](model.py) which contains the code to prepare the training data and generate and train the model
* [drive.py](drive.py) for driving the car in autonomous mode. It is nearly the original file with only few changes in the image preparation.
* [model.h5](model.h5) the trained model

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

### Model Architecture and Training Strategy

#### 1. The model architecture

 The model strongly follows the architecture of the nvidia paper <!-- TODO paper link --> with a few changes.  The first layer is for normalization. The following 1x1 convolutional layer follows the idea of let the model decide which is the best color space for the problem. The following layers are taken from the nvidia paper with relu activation and batch normalization after each convolutional layer. An additional dropout layer is added before the fully connected layers.

#### 2. Avoiding overfitting

One part of avoiding overfitting is simply generate enough training data. Secondly a dropout layer is used. With a validation set the model is tested for overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. The model was saved after each epoch an tested in the simulator to find the best number of epochs. The batch size for the training was chosen as 32.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
