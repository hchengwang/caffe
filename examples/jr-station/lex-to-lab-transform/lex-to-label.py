lex=open('../../data/vgg_data/data_set/lexicon.txt','rb')
#lab=open('../../data/vgg_data/labels.txt','rb')
f=open('lex-to-labels.txt','wb')
line = 0
index = 0
while True:
	index = 0	
	content = lex.readline()
	if content == "":
		break
	lab = open('../../data/vgg_data/labels.txt','rb')

	while True:
#		lab=open('../../data/vgg_data/labels.txt','rb')
#		lab.seek(0)
		label = lab.readline()

		if label == "":
			break
		if label == content:
			break
		index+=1
	lab.close()
	f.write(str(line)+" "+str(index)+"\n")	

	line += 1	
lex.close()

f.close()
