# ==============================
# Imports
# ==============================
import cv2
import numpy as np



# ==============================
# Square Image
# ==============================
def perfect_square(img):
    x = np.shape(img)
    
    diff = np.abs(x[0] - x[1])
    a = int(diff/2)
    
    if(len(x) == 3):
        if (x[0] < x[1]):
            b = int(x[1]-(diff/2))
            res = img[:,a:b,:]
        else:
            b = int(x[0]-(diff/2))
            res = img[a:b,:,:]
    else:
        if (x[0] < x[1]):
            b = int(x[1]-(diff/2))
            res = img[:,a:b]
        else:
            b = int(x[0]-(diff/2))
            res = img[a:b,:]
        
    return res



# ==============================
# Resize Image
# ==============================
def resize_128(img):
    return cv2.resize(img,(128,128), interpolation=cv2.INTER_AREA)


# ==============================
# Load Image
# ==============================
image = cv2.imread('dummy.png')
squared_image = perfect_square(image)
resized_image = resize_128(squared_image)

# ==============================
# Save Image
# ==============================
cv2.imwrite('dummy_resized.png', resized_image)



# ==============================
# Display Image
# ==============================
# import matplotlib.pyplot as plt
# plt.imshow(resized_image)
# plt.title(resized_image.shape[:2])
# plt.show()