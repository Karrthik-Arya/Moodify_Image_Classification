# Image Classification Model
## The Dataset
- We have used the fer2013 dataset and tensorflow for training the model

- This dataset contains images tagged with 7 emotions which we are going to work on grouping to 4. 

## The Baseline model 
- We tried using CNN to train the model.

- The model mainly involved tweaking and hit and trial to see which model architecture, hyperparameters worked the best. We explored the net, saw what other people had done, used that as a basis and created this model.

- We also used Dropout layers after every 2  CNN layers and between every FC layer. We have also added batch normalization layers after every 2 CNN layers. These helped to prevent our model from overfitting.

- Our convolution part consists of 4 pairs of convolutional layers→ {16, 16, 32, 32, 64, 64, 128, 128}

- Our dense network is a 4 layered network including the output layer.

- We tried keeping higher dropout rates, around 30-40 % but it didn’t give as satisfactory results as them being 25%

- We used Data augmentation by rotating random images horizontally … and also increased the number of epochs... this led to val_accuracy increasing from around 56 to 60% and above.
- We have used the adaptive Adam optimizer to help our model converge fast. The learning rate of about 0.0008 worked to give optimum results. We used a batch size of 64 and then ran the model for 120 epochs. 
- We have also used model checkpoints as a callback to get the best possible weights from the training.
Finally the model gave a val_accuracy of about 63 % and a test accuracy of about 64.22 % which was pretty good according to us since most of the models we found on the net did not have much better accuracy than this.  
                                                               
- This link was also useful https://github.com/AmrElsersy/Emotions-Recognition

## The Resnet50 model
- We found that many people had tried to use transfer learning for emotion recognition. We also thought of looking into that.We looked into the models covered in Facial Expression Recognition with Deep Learning paper.
 
- We then thought of going with the Resnet50 model. ResNet50 is a variant of the ResNet model which has 48 Convolution layers along with 1 MaxPool and 1 Average Pool layer. We tried using the pretrained model which was trained on the Imagenet dataset to classify objects. We used the CNN part of this pretrained model as a starting point. We added 3 FC layers after these CNN ones with units 1024, 256 and 7(the output layer).

- We observed  that unfreezing conv4 layers of block3 was important, without them the model does not converge. Other things that we tried were unfreezing all batch norm layers and then adding dropout between the FC layers. That seems to slightly improve the situation by preventing overfitting. Another thing that we tried was to use the SGD optimizer instead of Adam. This also gave slightly better results. With SGD the learning is much slower and thus we increased the learning rate to 0.01. Using SGD the model generalizes much better than using Adam or any other adaptive optimizers.  Now this model can get val/test accuracy about 60% after 60-70 epochs.(However still less than the baseline model)

- We are still working on this model to see if we can modify it to give better results. 


