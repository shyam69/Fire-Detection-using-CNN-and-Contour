import cv2
import tensorflow as tf
import numpy as np
def calcPercentage(msk): 
	height, width = msk.shape[:2] 
	num_pixels = height * width 
	count_white = cv2.countNonZero(msk) 
	#print(count_white)
	percent_white = (count_white/100000) * 100 
	
	percent_white = round(percent_white,2) 
	return percent_white 
# import the necessary packages
boundaries = [
	([17, 15, 100], [50, 56, 200]),
	([86, 31, 4], [220, 88, 50]),
	([25, 146, 190], [62, 174, 250]),
	([103, 86, 65], [145, 133, 128])
]
import cv2
x=0
y=0
CATEGORIES = ["Fire","No Fire"]
def prepare(file):
    IMG_SIZE = 50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
model = tf.keras.models.load_model("CNN.model")
image1 = input("ENTER IMAGE NAME: ") #your image path
prediction = model.predict(prepare(image1))
prediction = list(prediction[0])
print(prediction)
r=CATEGORIES[prediction.index(max(prediction))]
#print(r)
image = cv2.imread(image1)
if "Fire" in r :
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv,(10, 100, 20), (25, 255, 255) )
	
	x=calcPercentage(mask)
	#print(x)
	if x>100:
		x=100
	if x>20:
		print("AFFECTED BY FIRE")
	else:
		print("NORMAL")
	print("AFFECTED LEVEL: "+str(x))
	cv2.imshow("Fire", mask);cv2.waitKey();cv2.destroyAllWindows()
	
cv2.waitKey(0)
cv2.destroyAllWindows()
