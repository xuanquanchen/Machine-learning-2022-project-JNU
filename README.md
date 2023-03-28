# File construction


==**The two output logs of the Inception V3 implementation, model and dataset are uploaded to the cloud storage**==

You can get from https://www.aliyundrive.com/s/jMvxnVoijbr or https://1drv.ms/u/s!AkWL65N6AHvH4wAx0MPo9NyY7trs?e=y7ihHF

The retrained model without image augmentation is the output_graph.ph,  the labels are in output_labels.txt and the log of the training set and validation set is in the retrain_logs

The retrained model with image augmentation is the output_graph_aug.ph,  the labels are in  output_aug_labels.txt and the log of the training set and validation set is in the retrain_aug_logs



## Dataset folder

dataset folder contains the dataset we built



## Prepossessing.ipynb file

It contains the prepossessing part



## Part 1.ipynb file

It contains the steps of applying random forest to perform the classification and the steps to modify the parameters.



## Inception V3 part

### labelsImg.py file

It is the script to use the new model to do the classification task.

### test image folder

It contains the image used to test the new model.

### Original folder

It contains the output labels of the model without image augmentation

The retrained model is the output_graph.ph,  the labels are in output_labels.txt and the log of the training set and validation set is in the retrain_logs

#### retrain.py file

It is the modifiedretrain script, we add image augmentation in the main function of this file. If you want to remove it, remove the read_image function in the main file


### With image augmentation folder

It contains the output labels of the model with image augmentation

The retrained model is the output_graph_aug.ph,  the labels are in  output_aug_labels.txt and the log of the training set and validation set is in the retrain_aug_logs

#### retrain_with_aug.py file

It is the modified retrain script with the image augmentation, and the image augmentation part is in the imgAug.py

#### imgAug.py

This is file stored the function to perform image augmentation and used in retrain_with_aug.py
