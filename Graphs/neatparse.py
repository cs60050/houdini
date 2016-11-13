import matplotlib.pyplot as plt
from math import *
lines=[]
with open ('sample.txt','rt') as in_file:
	for line in in_file:
		lines.append(line)

#print(lines)
score=[]
for line in lines:
	score.append(line.split(" "))

#print score
lenght=len(score)

i=0

generation=[]
best_fitness=[]
avg_fitness=[]

while i < lenght :
		if i%16==1:
			
			generation.append(score[i-1][4])

	
		if i %16==4:
			avg_fitness.append(score[i-1][3])
		if i%16 ==5:
			best_fitness.append(score[i-1][2])
		i+=1
plt.plot(generation,avg_fitness)
plt.ylabel("Average Fitness --->")
plt.xlabel("No. of Generations --->  ")
plt.show()
