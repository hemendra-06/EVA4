###### Team members:
    - Umesh Singh (mshsingn772@gmail.com)
    - Hemendra Srinivasan (hemendra1111@gmail.com)


- The model achieves an accuracy of 81% with 35 epochs under the given contraints of parameters less than 1M.

- The model includes dilation in convblock2 and Depthwise separable convolution in convblock3. The test data normalisation includes RandomHorizontalFlip and RandomCrop

- The modularised code includes files:
	
    a. the model folder contains different model architectures like,

		- custom architecture for cifar10
		- ResNet18 architecture

	c. imagetransforms.py -

		provides the transformation for test and train data
	
	b. dataloader.py - 

		loads the train and test data with their correponding transformations
	
	d. train_test_model.py - 

		performs check for the available devices and provides function for running the model

	e. utils.py -
		contines the utility functions like display of model summary 
