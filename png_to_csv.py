#Converts a PNG file which is like a binar map of scene (only black and white pixels)

import cv2
import numpy as np
# from PIL import Image

# Load the image using OpenCV
image = cv2.imread("/home/abven/habitat-sim/examples/tutorials/nb_python/spawn_images/top_down_map.png", cv2.IMREAD_GRAYSCALE)
# print(image)
print(image.shape)

# image = Image.open("/home/pradnya/Documents/MP_Project/habitat/habitat-sim/examples/tutorials/Output/top_down_map.png", 'r')
# # Convert the image to binary (0 and 1)
binary_image = np.where(image > 0, 1, 0)

# # Get image dimensions
height, width = binary_image.shape
# print(binary_image.shape)

# # Reshape the 2D array to a 1D array (flattening)
# flat_binary_image = binary_image.reshape(-1)
# flat_binary_image = binary_image.flatten()

# # Save the 1D array as a CSV file
np.savetxt("/home/abven/habitat-sim/examples/tutorials/nb_python/binary_top_down_map.csv", binary_image, delimiter=',', fmt='%d')
# pix_val = list(image.getdata())
# # Print a message indicating successful completion
print("Conversion and saving to CSV completed successfully.")
# print("LIST OF PIXEL VALUES IN THE GIVEN IMAGE: ", pix_val)


