build/tools/caffe train --solver=examples/jr-station/solver.prototxt --weights=examples/jr-station/pre.caffemodel 2>&1 | tee log_-1.txt

python examples/jr-station/select_cluster_file.py 2000000
sh examples/jr-station/part_lmdb.sh
rm examples/jr-station/pre.caffemodel
cp examples/jr-station/vgg_iter_*.caffemodel examples/jr-station/caffemodel/0-2000000/
mv examples/jr-station/vgg_iter_*.caffemodel examples/jr-station/pre.caffemodel
build/tools/caffe train --solver=examples/jr-station/solver.prototxt --weights=examples/jr-station/pre.caffemodel 2>&1 | tee log_0.txt

python examples/jr-station/select_cluster_file.py 4000000
sh examples/jr-station/part_lmdb.sh
rm examples/jr-station/pre.caffemodel
cp examples/jr-station/vgg_iter_*.caffemodel examples/jr-station/caffemodel/2000000-4000000/
mv examples/jr-station/vgg_iter_*.caffemodel examples/jr-station/pre.caffemodel
build/tools/caffe train --solver=examples/jr-station/solver.prototxt --weights=examples/jr-station/pre.caffemodel 2>&1 | tee log_1.txt
