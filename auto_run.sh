sh examples/jr-station/part_lmdb.sh
build/tools/caffe train -solver examples/jr-station/solver.prototxt 2>&1 | tee log_0.txt

