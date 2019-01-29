from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# dimensions of our images
img_width, img_height = 220 , 90
model = load_model('test_model.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# predicting images
img = image.load_img('testing/img6.jpeg', target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
print (classes)

#predicting multiple images at once
#img = image.load_img('test2.jpg', target_size=(img_width, img_height))
#y = image.img_to_array(img)
#y = np.expand_dims(y, axis=0)

#images = np.vstack([x, y])
#classes = model.predict_classes(images, batch_size=10)

# print the classes, the images belong to
if classes[0][0] == 0 :
	prediction = 'NumberPlate'
	print(prediction)
else:
	prediction = 'No'
	print(prediction)
#print (classes[0])
#print (classes[0][0])