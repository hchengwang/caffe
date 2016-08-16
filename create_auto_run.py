f=open('auto_run.sh','wb')
for i in range(-1,61):
	f.write("python data/vgg_data/select_part.py "+str((i+1)*150000)+"\n")
	f.write("python examples/jr-station/lex-to-label-file.py\n")
	f.write("sh examples/jr-station/part_lmdb.sh\n")
	if i >= 0:
		f.write("mkdir examples/jr-station/caffemodel/"+str(i*150000)+"-"+str((i+1)*150000)+"\n")
		f.write("cp examples/jr-station/vgg_iter_*.caffemodel examples/jr-station/"+str(i*150000)+"-"+str((i+1)*150000)+"/\n")
		f.write("rm -f examples/jr-station/vgg_iter_*.solverstate"+"\n")
		f.write("mv examples/jr-station/vgg_iter_*.caffemodel examples/jr-station/pre.caffemodel\n")
	f.write("build/tools/caffe train --solver=examples/jr-station/solver.prototxt --weights=examples/jr-station/pre.caffemodel\n")
f.close()
