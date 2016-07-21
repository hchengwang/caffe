MY=examples/lab

echo "Create train lmdb.."
rm -rf $MY/img_train_lmdb
build/tools/convert_imageset --shuffle --resize_height=256 --resize_width=256 ~/caffe/data/lab_dataset/train/ $MY/train.txt $MY/img_train_lmdb

echo "Create test lmdb.."
rm -rf $MY/img_test_lmdb
build/tools/convert_imageset --shuffle --resize_height=256 --resize_width=256 ~/caffe/data/lab_dataset/test/ $MY/test.txt $MY/img_test_lmdb

echo "All Done.."


