import matplotlib.pyplot as plt
from math import *
lines=[]
with open ('out.txt','rt') as in_file:
	for line in in_file:
		lines.append(line)

#print(lines)
score=[]
for line in lines:
	score.append(line.split(" "))


lenght=len(score)

i=0
j=0
while i < lenght :
	
		temp=score[i][1].split(",")
		score[i][1]=temp[0]
		temp=score[i][4].split(",")
		score[i][4]=temp[0]
		temp=score[i][9].split("Q")
		score[i][9]=temp[0]
		temp=score[i][12].split("\n")
		score[i][12]=temp[0]
		
		i+=1

episode=[]
scores=[]
avg_frame=[]
Q=[]
i=0
a=0
b=0
c=0
j=0

print (lenght)
while j<=10:
	if i>= lenght:
		break
	a+=float(score[i][4])
	#print i
	b+=float(score[i][9])
	c+=int(score[i][12])
	if j==10:
		#print ('1')
		episode.append(score[i][1])
		scores.append((float(a/10)))
		avg_frame.append((float(b/10)))
		Q.append(float(c/10))
		a=0
		b=0
		c=0
		j=-1
		
	j+=1
	i+=1


	
#print (scores)

plt.plot(episode,avg_frame)
plt.ylabel("Avg.No. of Frames survived --->")
plt.xlabel("No. of Episodes --->  ")
plt.show()

