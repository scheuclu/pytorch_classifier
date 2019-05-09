# pytorch_classifier

This is a simple side-project to a bounding box detection project we are working on.

## Motivation

In the bounding box detection project, the data is quite noisy.
Looking at some ground truth bounding boxes, one can quickly see that there a quite some ground truth annotaions, that even a human would not recognize asbeing the labele grount truth class.

What is more, depending on which subset of labelled classes a network is trained on, there might be some very similar classes that the network now is forced to labels as "negative, no-object". This project also tries to quantify the significance of this problem.
