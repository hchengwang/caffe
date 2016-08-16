f=open("raw_part_train.txt","rb")
o=open("part_train.txt","wb")

while True:
	content = f.readline()
	if content == "":
		break
	for i in range(0,len(content)):
		if content[i] == " ":
			break
	path = content[:i]
	label=content[(i+1):]
	com = open("lex-to-labels.txt","rb")
	while True:
		pair = com.readline()
		if pair == "":
			break
		for i in range(0,len(pair)):
                	if pair[i] == " ":
                        	break
	        lex = pair[:i]
	        lab = pair[(i+1):]
		if lex == label[:-1]:
			break
	o.write(path+" "+lab)
	com.close()

f.close()
o.close()
