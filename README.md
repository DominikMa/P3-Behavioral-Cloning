# **Behavioral Cloning**

This repository contains my solution for the project "Behavioral Cloning" of the Udacity Self-Driving Car Engineer Nanodegree Program. The python code to generate the model could be found in [model.py](model.py).
More information on how to use the simulator can be found at [CarND-Behavioral-Cloning-P3](https://github.com/udacity/CarND-Behavioral-Cloning-P3)

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

[model]: ./model.png "Model Architecture"
[example_training_data]: ./images_writeup/example_training_data.png "Example training data"
[histo_raw_track1]: ./images_writeup/histo_raw_track1.png "Histogram of raw training data"
[histo_gauss0.005_track1]: ./images_writeup/histo_gauss0.005_track1.png "Histogram of training data with noise"
[histo_gauss0.005_reducezero0.08_track1]: ./images_writeup/histo_gauss0.005_reducezero0.08_track1.png "Histogram of training data with noise and reduced zero measurements"
[histo_gauss0.005_reducezero0.08_limit120_track1]: ./images_writeup/histo_gauss0.005_reducezero0.08_limit120_track1.png "Histogram of training data with noise, reduced zero measurements and a bin limit of 120"
[histo_gauss0.005_reducezero0.08_limit120_track1+2]: ./images_writeup/histo_gauss0.005_reducezero0.08_limit120_track1+2.png "Histogram of training data with noise, reduced zero measurements and a bin limit of 120 for track one and two"

### Included Files

This repository contains all required files to generate, train and test the model and can be used to run the simulator in autonomous mode. It does not contain the training data.

The required files are:
* [model.py](model.py) which contains the code to prepare the training data and generate and train the model
* [drive.py](drive.py) for driving the car in autonomous mode. It is nearly the original file with only few changes in the image preparation.
* [model.h5](model.h5) the trained model
* [output_video_track1.mp4](record_track1/output_video_track1.mp4) a video if the model driving through the first track

Additional the following files are included:
* [model.json](model.json) which contains the used parameters for training the model
* [model.png](model.png) a plot of the model
* [output_video_track2.mp4](record_track2/output_video_track2.mp4) a video if the model driving through the second track
* [videos](videos/) four videos of the model driving each track forward and backward

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

### General Model Architecture and Training Strategy

#### 1. The model architecture

 The model strongly follows the architecture of the [nvidia paper](https://arxiv.org/pdf/1604.07316v1.pdf) with a few changes.  The first layer is for normalization. The following 1x1 convolutional layer follows the idea of let the model decide which is the best color space for the problem. The following layers are taken from the nvidia paper with relu activation and batch normalization after each convolutional layer. An additional dropout layer is added before the fully connected layers.

#### 2. Avoiding overfitting

One part of avoiding overfitting is simply generate enough training data. Secondly a dropout layer is used. With a validation set the model is tested for overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. The model was saved after each epoch an tested in the simulator to find the best number of epochs. The batch size for the training was chosen as 32.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. To get training data the simulator was. It was tried to drive in the center of the road. For track one there were multiple rounds captured forward, backward and on different graphic settings. For the second track only forward training data was generated to see if the model is able to generalize.



### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Since using the model architecture presented in the udacity course as a starting point did not lead to an useful solution, the next step was to try the nvidia model. This architecture already results in a good first model. This architecture was the modified by adding batch normalization, dropout and an additional first layer.

The generated data was split into a training and validation set. The model was then trained, validated and saved after each epoch. It was observed that the model converges pretty fast and gives good results for the train and test error.

Unfortunately at this point a low train or test error does not mean that the car drives well in the simulator.

The reason for this was the distribution over the training data. This issue will be described later in detail.

A better preparation of the training data then results in a model which was able to drive safely through the first track at full speed.
It also was able to drive the track backwards which is not surprising because there were some training data of the track backwards.

After this the model was trained again from scratch with the additional training data of track two. Here the model was trained for 10 epochs, some parameters were fine tuned and the model was again train for about 30 epochs. The resulting model was able to drive around track one forward and backward at full speed and around track two forward and backward at 11 MPH.
It was able to drive track two backwards despite it has never seen training data with the track backward.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

![model architecture][model]

The first lambda layer does a normalization and mean centering.

The architecture has 253,175 parameters in total from which 252,703 are trainable.


#### 3. Creation of the Training Set & Training Process
While choosing the architecture was straight forward getting good training data was quite difficult.

To capture good driving behavior, first four laps forward on track one for the the graphic settings fastest, fast, simple and beautiful each were recorded. Then the same backwards two labs. Here are example images of center lane driving:

![example training data][example_training_data]


To augment the data set the images and angles are flipped, resulting in 231960 training images.

After the collection process, the training data consists of 115980 images for the first track. These images are converted to the HSV color space, cropped 60px of the top and 25px on the bottom and the resized to 200x66.

Since using this training data does not result in a good driving model in had a deeper look on the data distribution.
As one can see in the histogram of the training data the distribution does not look good for training. There are a lot of measurements which are zero and because the side images of these are used there are two additional spikes. Furthermore there are many small spikes. This is because the gamepad does not really gives continuous values but values with a fixed resolution.

![histogram of raw training data][histo_raw_track1]

Therefor in a first step some random gaussian noise is added to the measurements, resulting in the following histogram.

![histogram of training data with noise][histo_gauss0.005_track1]

This results in a way more smooth distribution of the measurements, but there are still three spikes.
Using data with measurements of zero only with a probability of 0.08 will reduce these. Additionally we use the side images only with a probability of 0.4.

![histogram of training data with noise and reduced zero measurements][histo_gauss0.005_reducezero0.08_track1]

We can now see a smooth distribution. After this there are about 102000 training images left. Still we notice that measurements between -0.15 and 0.15 are frequent while others are rare. Using this training data the model will get a low error by simply outputting only measurements in this range, meaning there is no need of learning the rare measurements.
Because we do not want our model to be biased about the measurement we need a more uniform distribution.
To archive this we sort the measurements in bins of the size 0.005 and then limit the size of each bin to 120. Doing this gives the following result with 2600 training samples:

![histogram of training data with noise, reduced zero measurements and a bin limit of 120][histo_gauss0.005_reducezero0.08_limit120_track1]

This makes the rare samples more important.
Using the training data from track one and two we got

![histogram of training data with noise, reduced zero measurements and a bin limit of 120 for track one and two][histo_gauss0.005_reducezero0.08_limit120_track1+2]

which looks way better for training.

Training first with the whole data set and then fine tune the model with the data set with limited bin size gives the best result.
