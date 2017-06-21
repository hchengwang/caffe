# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import time
# display plots in this notebook
#%matplotlib inline

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

import os
if os.path.isfile(caffe_root + 'data/vgg_data/dictnet_vgg.caffemodel'):
    print 'VGG Models found.'
else:
    print 'Downloading pre-trained VGG models...'
#    !wget vgg_models.tar.gz -O https://www.dropbox.com/s/v1avmkm7q2i6d1e/vgg_models.tar.gz?dl=0
 #   !tar xvf vgg_models.tar.gz
caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()
# caffe.set_mode_cpu()

model_def = caffe_root + 'data/vgg_data/dictnet_vgg_deploy.prototxt'
model_weights = caffe_root + 'examples/jr-station/pre.caffemodel'
print model_def
print model_weights

net = caffe.Net(model_def,      # defines the structure of the model#
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# for each layer, show the output shape
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(1,        # batch size
                          1,         # 1-channel gray images
                          32, 100)  # image size is 32x100
in_data = open('jr-station/vgg_test.txt','rb')
#out_data = open('images/2016-06-28/all_predict.txt','wb')

count = 0
total = 0
while True:
    
    content = in_data.readline()
    if content == "":
        break
    #read filename
    for i in range(0, len(content)):
        if content[i] == " ":
            break
    total += 1
    path = content[:i]
    label = content[(i+1):-1]
    
    img_path = caffe_root + 'data/vgg_data/data_set/' + path
    image_rgb = caffe.io.load_image(img_path)
    transformed_image = transformer.preprocess('data', image_rgb)
    plt.imshow(image_rgb)

    image = caffe.io.load_image(img_path, False)
    transformed_image = transformer.preprocess('data', image)
    # plt.imshow(image)

    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    pred = output['prob'][0].argmax()
    
    labels_file = caffe_root + 'data/vgg_data/labels.txt'
    if not os.path.exists(labels_file):
        print 'label file does not exist'

    labels = np.loadtxt(labels_file, str, delimiter='\t')    
    
    if labels[pred].upper() == label.upper():
        count += 1
        print "Hit!"
    if count % 1000 == 0 and count > 0:
        print count
    
    
    print "label: "+labels[pred]+" pred: " + str(pred) + " path:"+ path


        
    print "accurarcy:"
    print float(count)/total
    
    
    
