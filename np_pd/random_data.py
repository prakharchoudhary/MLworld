from random import randint
f = open("web_traffic.tsv", "w+")

for i in range(1, 1000):
	if i<500:
		x = "%d\t%d\n"%(i, randint(i, 200+i))

	f.write(x)

f.close()