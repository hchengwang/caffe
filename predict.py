import numpy as np
import matplotlib.pyplot as plt
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
if os.path.isfile(caffe_root + 'models/vgg_dictnet/dictnet_vgg.caffemodel'):
    print 'VGG Models found.'
else:
    print 'Downloading pre-trained VGG models...'
    !wget vgg_models.tar.gz -O https://www.dropbox.com/s/v1avmkm7q2i6d1e/vgg_models.tar.gz?dl=0
    !tar xvf vgg_models.tar.gz

caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()
# caffe.set_mode_cpu()

model_def = caffe_root + 'models/vgg_dictnet/dictnet_vgg_deploy.prototxt'
model_weights = caffe_root + 'models/vgg_dictnet/dictnet_vgg.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(1,        # batch size
                          1,         # 1-channel gray images
                          32, 100)  # image size is 32x100

for i in range(1, 105):
	img_path = caffe_root + 'examples/images/tag_text/0_FOUR/FOUR_'+str(i)+'.jpg'

	image_rgb = caffe.io.load_image(img_path)
	transformed_image = transformer.preprocess('data', image_rgb)
	plt.imshow(image_rgb)

	image = caffe.io.load_image(img_path, False)
	transformed_image = transformer.preprocess('data', image)
	# plt.imshow(image)

	# copy the image data into the memory allocated for the net
	net.blobs['data'].data[...] = transformed_image

	### perform classification
	output = net.forward()

	output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

	print 'predicted class is:', output_prob.argmax()

	# load ImageNet labels
	labels_file = caffe_root + 'models/vgg_dictnet/dictnet_vgg_labels.txt'
	if not os.path.exists(labels_file):
	    print 'label file does not exist'
	    
	labels = np.loadtxt(labels_file, str, delimiter='\t')

	print 'output label:', labels[output_prob.argmax()]