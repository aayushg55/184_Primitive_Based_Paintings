import numpy as np

# Example 2D array with shape (2, n)
array_2d = np.array([[1, 2, 3],
                     [4, 5, 6]])

arr = np.ones((2,9539))

# Example 1D boolean mask with shape (n,)
mask = np.zeros(9539, dtype=bool)

print(mask.shape, array_2d.shape)
# Index the 2D array using the mask
masked_array = arr[:, mask]

# Print the results
print("Original 2D array:")
print(array_2d)
print("\nMask array:")
print(mask)
print("\nMasked array:")
print(masked_array)