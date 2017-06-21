f=open('auto_run.sh','wb')

n = 2000000
for i in range(-1,5):
	f.write("python examples/jr-station/select_cluster_file.py "+str((i+1)*n)+"\n")
	f.write("sh examples/jr-station/part_lmdb.sh\n")
	if i >= 0:
		f.write("mkdir examples/jr-station/caffemodel/"+str(i*n)+"-"+str((i+1)*n)+"\n")
		f.write("cp examples/jr-station/vgg_iter_*.caffemodel examples/jr-station/caffemodel/"+str(i*n)+"-"+str((i+1)*n)+"/\n")
		f.write("rm -f examples/jr-station/vgg_iter_*.solverstate examples/jr-station/pre.caffemodel"+"\n")
		f.write("mv examples/jr-station/vgg_iter_*.caffemodel examples/jr-station/pre.caffemodel\n")
	f.write("build/tools/caffe train --solver=examples/jr-station/solver.prototxt --weights=examples/jr-station/pre.caffemodel 2>&1 | tee log_"+str(i)+".txt\n\n")
f.close()
