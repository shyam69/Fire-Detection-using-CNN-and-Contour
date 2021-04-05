import cv2
import tensorflow as tf
import numpy as np
x=0
y=0
CATEGORIES = ["No Fire","Fire"]
def prepare(file):
    IMG_SIZE = 50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
model = tf.keras.models.load_model("CNN.model")
image = "1.jpg" #your image path
prediction = model.predict(prepare(image))
prediction = list(prediction[0])
print(prediction)
r=CATEGORIES[prediction.index(max(prediction))]
print(CATEGORIES[prediction.index(max(prediction))])
if 'No Fire' in r:
	gray = cv2.imread(image, cv2.COLOR_BGR2GRAY)
	white_bg = 255*np.ones_like(image)
	ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
	blur = cv2.medianBlur(thresh, 1)
	kernel = np.ones((10, 20), np.uint8)
	img_dilation = cv2.dilate(blur, kernel, iterations=1)
	ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
	for i, ctr in enumerate(sorted_ctrs):
		# Get bounding box
		x, y, w, h = cv2.boundingRect(ctr)
		roi = image[y:y + h, x:x + w]
		if (h > 10 and w > 10) and h < 200:
			cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)        
			cv2.imshow('{}.png'.format(i), roi)
			white_bg[y:y+h, x:x+w] = roi
cv2.imshow('Rcnn', image)
