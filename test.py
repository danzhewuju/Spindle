import threading
import time
import Levenshtein
import pandas as pd
a = "022345"
b = "012345"
print(Levenshtein.jaro(a, b))