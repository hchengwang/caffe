f = open('train.txt','r')
o = open('out_test.txt','w')
i=0
while True:
	content = f.readline()
	if content == "":
		break
	i = i + 1
#	o.write(content[2:])
f.close()
o.close()
print i
