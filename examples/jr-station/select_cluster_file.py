import sys
f=open('examples/jr-station/train.txt','rb')
o=open('examples/jr-station/part_train.txt','wb')
f1=open('examples/jr-station/train.txt','rb')
o1=open('examples/jr-station/part_test.txt','wb')
line = 0
while True:
	if line > int(sys.argv[1])-200000:
		o1.write(content)
	if line >= int(sys.argv[1]):
		break
	
	content = f.readline()
	if content == "":
		break
	line = line + 1
while True:
	content = f.readline()
	if content == "":
		break
	if line >= (int(sys.argv[1])+2000000):
		break
	
	line = line + 1
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
#o1.close()


