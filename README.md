# slope-predict

A final project for COMP 562 - Machine Learning

Merha B, Shu N, Sahith R, Jason X

![Data Collection](slope-predict/blob/main/figures/DataCollection.png)

## Motivation

Smartphones have become so ubiquitous in our lives that we carry them everywhere in our pockets. We rely on phones to conduct everyday tasks, such as tracking our health. Phones today are able to count steps, measure running/walking distance, and even evaluate gait symmetry. This project aims to take the level of health tracking a step further. 

Imagine that you are on the threadmill. You turn the slope on high to really burn those quads, but the phone in your pocket is only counting steps as if you are walking on flat ground. 

That's not very satisfying. Your phone should be able to accurately track how hard you worked by accounting for walking slope, so that what we set out to do. In this project, we propose a machine learning model to predict ground slope using phone accelometer and gyroscope data.

## Methods

A smartphone was attached to a participant's right leg. As the participant walks on a treadmill with various inlines, the smartphone collects motion data while another device measures the ground slope. Nine features were measured: x,y,z acceleration, x,y,z angular velocity, azimuth, pitch, and roll. 

We used logistic regression on feature scaleograms to determine the baseline performance and identify the best features to use. It was found that the y-acceleration, x-angular velocity, and pitch had the most predictive power. We then trained a logistic regression model using the scaleograms of the selected features. Separately, a 1D convolutional neural network was trained with the selected features using augmented data. 
