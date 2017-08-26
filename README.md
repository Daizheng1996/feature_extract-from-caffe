# feature_extract-from-caffe
Use python script to extract feature from trained caffe-net,images and file-label pair text in the form of 1-D array.
----------------------------------------------------------------------------------------------

Sometimes after you train a caffe net,you probably want to extract features from different layers and feed them to classifiers such as SVM or RF, but the built-in tools of caffe for feature_extraction is hard to use.Thus this python script is developed to help you extract feature from caffemodel,original images,image-label-pair-text,and save the feature and label in the form of npy file(which can be load by numpy function numpy.load() as a form of nd-array),and then you can use sklearn.svm to train a classifier use the feature and labels you extracted.

----------------------------------------------------------------------------------------------
FILE NEEDED:
a already trained caffe-model
the deploy file of the caffe-model
the original image of your dataset
the image-label pair text file

-----------------------------------------------------------------------------------------------
Dependency:
caffe,numpy
