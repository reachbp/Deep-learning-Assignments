import os
files = [f for f in os.listdir('.') if os.path.isfile(f)]
out = open("neg", "w")
for f in files:
	stuff = open(f, 'r').read()
	out.write("neg, " + stuff + '\n')
