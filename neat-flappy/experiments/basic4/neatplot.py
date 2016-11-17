# Author: Sohom Chakraborty

import matplotlib.pyplot as plt
from math import *
import sys

lines=[]
with open (sys.argv[1],'rt') as in_file:
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
		if i%14==0:
			# print "generation", score[i]
			generation.append(score[i][4])
		if i%14==2:
			# print "avg", score[i]
			avg_fitness.append(score[i][3])
		if i%14 ==3:
			# print "best", score[i]
			best_fitness.append(score[i][2])
		i+=1

print len(avg_fitness)
print len(best_fitness)
plt.plot(generation, avg_fitness, '-b', label='Average')
plt.plot(generation, best_fitness, '-g', label='Best')

plt.ylabel("Fitness --->")

plt.xlabel("No. of Generations --->  ")
plt.legend(loc='best')
plt.savefig(sys.argv[1])
