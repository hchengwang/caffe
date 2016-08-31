import sys
f=open('examples/jr-station/mini_dataset.txt','rb')
o=open('examples/jr-station/part_train.txt','wb')
#f1=open('examples/jr-station/train.txt','rb')
o1=open('examples/jr-station/part_test.txt','wb')
line = 0
while True:
	content = f.readline()
	if content == "":
		break
	line+=1

	if line <= 80000:
		o1.write(content)
	else:
		o.write(content)
	


	#o.write(content)
#while True:
#	content = f.readline()
#	if content == "":
#		break
#	if line >= (int(sys.argv[1])+2200000):
#		break
#	
#	line = line + 1
#	o1.write(content)
f.close()
o.close()
o1.close()


