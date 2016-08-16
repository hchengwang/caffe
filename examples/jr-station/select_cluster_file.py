import sys
f=open('../../data/vgg_data/train.txt','rb')
o=open('raw_part_train.txt','wb')
line = 0
while True:
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
	if line > (int(sys.argv[1])+150000):
		break
	line = line + 1
	o.write(content)
f.close()
o.close()


