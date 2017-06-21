#f1 = open("5000_list.txt","rb")
f2 = open("train.txt","rb")
o = open("mini_dataset.txt","wb")
j = 0
while True:
#	j+=1
#	if j <= 2380898:
#		continue
	content = f2.readline()
	if content == "":
		break
	fg = 0
	fg1 = 0
	fg2 = 0
	for i in range(0,len(content)):
		if content[i] == "_":
			if fg1 == 0:
				fg1 = i+1
			else:
				fg2 = i
				break
	label = content[fg1:fg2]
	label = label.lower()
	if label[0] < "e":
		s = 1
		e = 2358
	elif label[0] < "l":
		s = 2359
		e = 4085
	elif label[0] < "s":
		s = 4086
		e = 6363
	else:
		s = 6364
		e = 8181
#	print "out: "+label[0]
	f1 = open("10000_list.txt","rb")
	fg = -1
	line = 0
	while True:
#		print line
		ref = f1.readline()
#		print ref[:-2] + " " + str(len(ref)) 
		if ref == "":
			break

		line += 1
		if line < s:
			continue
		elif line > e:
			break
#
#		print ref[:-2] + " " + label[:-2] + " " + str(s) + " " +str(e)		
#		print str(len(ref[:-2]))+ " " + str(len(label[:-2]))
		if ref[:-2] == label[:-2]:
			fg = 0
#			print "haaaaaaaaaaaaaaaaaa"
			break
	if fg == 0:
		o.write(content) 	
	f1.close()
	if j%10000 == 0:
		print "j:"+str(j)
	j+=1
f2.close()
o.close()
