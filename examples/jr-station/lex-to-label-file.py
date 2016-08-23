f=open("examples/jr-station/raw_train.txt","rb")
o=open("examples/jr-station/train.txt","wb")

while True:
	content = f.readline()
	if content == "":
		break
	for i in range(0,len(content)):
		if content[i] == " ":
			break
	path = content[:i]
	label=content[(i+1):]
	com = open("examples/jr-station/lex-to-labels.txt","rb")
	la = int(label)
	while la >=0:
		pair = com.readline()
		la -= 1
	
	for i in range(0,len(pair)):
                if pair[i] == " ":
                        break

	lab = pair[(i+1):]

	o.write(path+" "+lab)
	com.close()

f.close()
o.close()
