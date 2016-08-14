MY=examples/jr-station

echo "Create part lmdb.."
rm -rf $MY/img_part_test_lmdb
build/tools/convert_imageset  --resize_height=32 --resize_width=100 ~/caffe/data/vgg_data/data_set/ $MY/part.txt $MY/img_part_test_lmdb

echo "All Done.."


