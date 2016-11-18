import numpy as np
import pickle
import os

# create method for writing output file given attention weights
# french_position-english_position

# check number of examples matches expected

translation_direction = "f2e" #"e2f"
attention_weights = np.array([[0.1,1],[1,3,4,2]])

for weight in attention_weights:
    print(np.argmax(weight))

"""
f = open("output.txt", 'w', encoding = "ISO-8859-1")
for weight in attention_weights:
    f.write("Example: "+noun+" guess="+" drug "+" gold= confidence="+str(10)+"\n")
f.close()
"""
