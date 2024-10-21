from collections import Counter
import numpy as np
cat_data1 = np.load('train.npz')
cat_data2 = np.load('valid.npz')
cat_data3 = np.load('test.npz')

# numbers = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]

list_sum = list(cat_data1["num_atoms"]) + list(cat_data2["num_atoms"]) + list(cat_data3["num_atoms"])
count = Counter(list_sum)

print(count)
total_count = sum(count.values())
print(total_count)
