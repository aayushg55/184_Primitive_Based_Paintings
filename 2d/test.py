import numpy as np

# Example 2D array with shape (2, n)
array_2d = np.array([[1, 2, 3],
                     [4, 5, 6]])

# Example 1D boolean mask with shape (n,)
mask = np.array([True, False, True])

print(mask.shape)
# Index the 2D array using the mask
masked_array = array_2d[:, mask]

# Print the results
print("Original 2D array:")
print(array_2d)
print("\nMask array:")
print(mask)
print("\nMasked array:")
print(masked_array)