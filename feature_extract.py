import os  
import sys
import caffe
import numpy as np 

caffe_root='/home/zhuyaochen/caffe/' #CHANGE IT TO THE ROOT OF YOUR CAFFE
deploy=caffe_root+'examples/DImage/deploy.prototxt' #CHANGE IT TO THE PATH OF YOUR PROTOTXT FILE
caffe_model=caffe_root + 'examples/DImage/caffenet_train_iter_1000.caffemodel'  #CHANGE IT TO THE PATH OF YOUR CAFFE MODEL
mean_file = caffe_root + 'examples/DImage/DImage_mean.binaryprot.npy' #CHANGE IT TO THE BINARYFILE 
synset_file = caffe_root + 'examples/DImage/synset_words.txt'#CHANGED IT TO THE PATH OF YOUR IMAGE-LABEL PAIR

def read_image(img_dir,data_name,abs = False):
    ###
    # img_dir is the fold name of your images
    # data_name is the text file that contains the name and the label of your images
    # this function seperate the name and the label and return them

    file = open(data_name,"r")
    str_filetitle = file.read()

    #Get the file names
    file_list = np.array(str_filetitle.split())[::2]
    #Get the label names
    label_list = np.array(str_filetitle.split())[1::2]

    if not img_dir.endswith('/'):
        img_dir += "/"

    file_list_total = []

    #Change the file names into absolute path
    if not abs:
        for i in range(len(file_list)):
            file_list_total.append(img_dir + file_list[i])

    return file_list_total,label_list


def feature_extract(img_list,layer_name,foldname = ""):  

    print 'extract feature from',layer_name
    net = caffe.Net(deploy,caffe_model,caffe.TEST)  

    #to change the shape
    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})  
    transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))  
    transformer.set_transpose('data', (2,0,1))  
    transformer.set_channel_swap('data', (2,1,0))  
    transformer.set_raw_scale('data', 255.0)

    num_of_image = len(img_list)

    #the number of images process in one iteration
    batchsize = 50
    #the time of iteration with a full batchsize
    iter_times = num_of_image / batchsize
    #reshape the data blob
    net.blobs['data'].reshape(batchsize,3,227,227)

    #the size of the feature of the layer you want to extract
    size = net.blobs[layer_name].data[0].size
    totel_size = size * num_of_image
    #create the list for features
    feature_list = np.arange(totel_size,dtype = float).reshape((num_of_image,size))

    #to show the current postion to input a feature
    cursor = 0
    for forward_time in range(iter_times):
    #the time of iteration
        for i in range(batchsize):
        #the number of images one iteration should process
            img = caffe.io.load_image(img_list[i + cursor])
            net.blobs['data'].data[i,:,:,:] = transformer.preprocess('data', img)
        #run a forward pass
        net.forward()
        #get the feature
        feature_list[cursor:cursor + batchsize,:] = net.blobs[layer_name].data.reshape((batchsize,size))
        cursor += batchsize

    #process the remain image which is less than the number of a batch
    image_rest = img_list[cursor:]
    num_of_rest = len(image_rest)
    net.blobs['data'].reshape(num_of_rest,3,227,227)
    for i in range(num_of_rest):
        img = caffe.io.load_image(image_rest[i])
        net.blobs['data'].data[i,:,:,:] = transformer.preprocess('data',img)
    net.forward()
    feature_list[cursor:,:] = net.blobs[layer_name].data.reshape((num_of_rest,size))
    #to save the feature as the form of npy(the form of ndarray)
    #if you want to open a npy file
    #the usage is:
    #import numpy as np
    #np.load('XXX.npy')

    if not foldname.endswith('/'):
        foldname += '/'
    np.save(foldname + 'feature_'+layer_name+'.npy',feature_list)

  


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] in ['help','-help','--help']:
        print "usage:"
        print "python + feature_extract.py + img_dir + data_dir + foldname + layername"
    else:
        if len(args) < 4:
            raise Exception('lack of paramete!')
        else:
            img_dir = args[0]
            data_dir = args[1]  #'/home/zhuyaochen/sample/data.txt'
            file_list,label_list = read_image(img_dir,data_dir)
            foldname = args[2]
            layer_name = args[3]
            feature_extract(file_list, layer_name, foldname)
            if not foldname.endswith('/'):
                foldname += '/'
            np.save(foldname+ 'label.npy',np.array(label_list))
