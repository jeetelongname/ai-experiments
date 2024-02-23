#!/usr/bin/env python3

# question 1, load the data set.
import pickle
import matplotlib.pyplot as plt

# here we read in the first batch as f
with open("data_batch_1.pkl", "rb") as f:
    data = pickle.load(f, encoding = "latin1")

# we reshape the images from a vector however long into a 32 x 32 x 3 image
images = data['data'].reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)

print("example template press q to exit")
template_index = 10
template = images[template_index]
plt.imshow(template)
plt.title(f'Template Example (Index: {template_index})')
plt.show()

# I have not attempted question 3 of part 5
