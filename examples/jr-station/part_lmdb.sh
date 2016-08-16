MY=examples/jr-station

echo "Create train lmdb.."
rm -rf $MY/img_part_train_lmdb
build/tools/convert_imageset  --gray=true --resize_height=32 --resize_width=100 ~/caffe/data/vgg_data/data_set/ $MY/part_train.txt $MY/img_part_train_lmdb

echo "Create test lmdb.."
rm -rf $MY/img_part_test_lmdb
build/tools/convert_imageset  --gray=true --resize_height=32 --resize_width=100 ~/caffe/data/vgg_data/data_set/ $MY/part_test.txt $MY/img_part_test_lmdb

echo "All Done.."


