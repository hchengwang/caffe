f=open('10000_list.txt',"rb")
leng = 0
while True:
	content = f.readline()
	if content == "":
		break
	if len(content) > leng:
#		print content
		leng = len(content)
	if len(content) > 10:
		print ">10:"+content[:-1]
print leng
f.close()
