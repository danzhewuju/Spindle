import threading
import time
import Levenshtein

a = ["1", "2", "3"]
b = [1, 2, 3]
c = dict(zip(a, b))
c = sorted(c.items(), key=lambda x: -x[-1])
print(c)
print(c[0][0])
