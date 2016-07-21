f1 = open('predict_0.txt','rb')
f2 = open('predict_1.txt','rb')
ac = open('accuracy_1.txt','wb')

Ft = Fh = At = Ah = St = Sh = Et = Eh = 0
while True:
	content = f2.readline()
	if content == "":
		break
	if content[0] == '0':
		Ft = Ft +1
		s = content[-5:-1]
		if s.upper() == "FOUR":
			Fh = Fh+1
	if content[0] == '1':
		St = St +1
		s = content[-6:-1]
		if s.upper() == "SCORE":
			Sh = Sh+1
	if content[0] == '2':
		At = At +1
		s = content[-4:-1]
		if s.upper() == "AND":
			Ah = Ah+1
	
	if content[0] == '3':
		Et = Et +1
		s = content[-6:-1]
		if s.upper() == "SEVEN":
			Eh = Eh+1

ac.write('FOUR: '+str(Ft)+'(t) hit: '+str(Fh)+'(h) '+str(float(Fh)/Ft)+'(a)\n')
ac.write('SCORE: '+str(St)+'(t) hit: '+str(Sh)+'(h) '+str(float(Sh)/St)+'(a)\n')
ac.write('AND: '+str(At)+'(t) hit: '+str(Ah)+'(h) '+str(float(Ah)/At)+'(a)\n')
ac.write('SEVEN: '+str(Et)+'(t) hit: '+str(Eh)+'(h) '+str(float(Eh)/Et)+'(a)\n')
