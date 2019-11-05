import sys

tot = 0
for line in sys.stdin:
    tkns = line.strip().split()
    tot += int(tkns[4]) - int(tkns[3])
print(tot)
