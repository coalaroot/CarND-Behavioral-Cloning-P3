**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/udacity_example.jpg "Udacity Example"
[image2]: ./examples/recovery_start.jpg "Recovery Start"
[image3]: ./examples/recovery_middle.jpg "Recovery Middle"
[image4]: ./examples/recovery_end.jpg "Recovery End"
[image5]: ./examples/recovery_end.jpg "Normal Image"
[image6]: ./examples/flip.jpg "Flipped Image"

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is exactly Nvidia architecture for self-driving cars [link](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)
It consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 61-65) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 59).

#### 2. Attempts to reduce overfitting in the model

To reduce overfitting the model is learning only for 2 epochs. Found out that more epochs is just redundant. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 74).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used Udacity provided data plus recorded a combination of center lane driving, recovering from the left and right sides of the road and some additional bridge crossing. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with what works (Nviadia architecture) and finetune the hyperparameters and create appropriate data from there.

I used that model because it was already tested by experts and showed great results.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80% training, 20% validation).

The final step was to run the simulator to see how well the car was driving around track one. It did't go through the first curve. Then I added some recovery data on all curves. This let me pass the first curve but it got stuck on the bridge. I added some data about that too. Surprisingly, I had to add way more data driving around the bridge (also driving clock-wise) that recovering from curves.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes

|Layer  | Details |
|:-----:|:----:|
|Normalization|Using x / 255 - 0.5|
|Cropping|Cropped 60px from the top and 25 from the bottom|
|Convolution|kernel 5x5, 24 layers|
|RELU||
|Convolution|kernel 5x5, 36 layers|
|RELU||
|Convolution|kernel 5x5, 48 layers|
|RELU||
|Convolution|kernel 3x3, 64 layers|
|RELU||
|Convolution|kernel 3x3, 64 layers|
|Flatten||
|Fully connected| output 100|
|Fully connected| output 50|
|Fully connected| output 10|
|Fully connected| output 1|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first just used Udacity provided data. Here's an example image

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to center of the road after wandering off to the side. These images show what a recovery looks like. The first image is when I started recording, middle of the recovery and the end of it:

![alt text][image2]
![alt text][image3]
![alt text][image4]

To augment the data set, I flipped images. Here's an example:

![alt text][image5]
![alt text][image6]

I used all three cameras from the simulation and added/subtracted 0.1 correlation respectively to the angles of the left/right camera image. 

After the collection process, I had 8742 number of images per simulation shot. Multiply that by 3 because we have 3 cameras on the car 8742 * 3 = 26226. The we flip every image so again multiply by 2, which is 26226 * 2 = 52452.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as overfitting just grew after that. I used an adam optimizer so that manually training the learning rate wasn't necessary.
