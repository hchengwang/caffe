#all_list = open('examples/images/list.txt','rb')
new_list = open('examples/images/2_list.txt','wb')
trans = open('examples/images/new_list_2.txt','rb')

while True:
	content = trans.readline()
	if content == "":
		break

	j=0
	while True:
		j=j+1
		if content[j]==' ':
			break

	tmp=content[:j] #name
	if content[2] == 'F':
		bonus = 104
	elif content[2] == 'A':
		bonus = 98
	elif content[3] == 'E':
		bonus = 59
	else:
		bonus = 39
	bonus = 0

	if tmp[-6] == '_': #2position
		print "haha1"
		tmp = tmp[:2]+str(int(tmp[-5:-4])+bonus).zfill(3)+'.jpg'
	elif tmp[-7] == '_': #2position
		print "haha2"
		tmp = tmp[:2]+str(int(tmp[-6:-4])+bonus).zfill(3)+'.jpg'
	else:
		print "haha3"
		tmp = tmp[:2]+str(int(tmp[-7:-4])+bonus).zfill(3)+'.jpg'
	print len(tmp)

	new_list.write(tmp+content[j:])


#all_list.close()
new_list.close()
trans.close()

