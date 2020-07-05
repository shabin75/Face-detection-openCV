
import cv2
import matplotlib.pyplot as plt

test_image = cv2.imread('1.jpg')

test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

plt.imshow(test_image_gray, cmap='gray')
plt.show()


def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor=1.2, minNeighbors=5);

print('Faces found: ', len(faces_rects))

for (x,y,w,h) in faces_rects:
     cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 5)


plt.imshow(convertToRGB(test_image))
plt.show()