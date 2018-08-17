import cv2
from keras.models import load_model
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import operator
import time
import vlc
import subprocess
import random
classifier=load_model("fer2013_mini_XCEPTION.110-0.65.hdf5")
camera=cv2.VideoCapture(0)

i=0
while True:
	ret,frame=camera.read()#ret=return,img=img data
	cv2.imwrite("abc.png",frame)
	cv2.waitKey(20)
	face=cv2.CascadeClassifier("/home/ankurgupta9352/Downloads/techienest_manit/haarcascade_frontalface_default.xml")
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	gray_resize=cv2.resize(gray,(64,64))
	y1=image.img_to_array(gray_resize)
	y1= np.expand_dims(y1, axis=0)
	y1=y1/255
	result=classifier.predict(y1)
	b=result.ravel()
	x=np.argmax(b)
	b1=["anger","disgussed","fear","happy","sad","surprise","neutral"]
	b2=b1[x]
	faces=face.detectMultiScale(gray,1.3,9)#1.3 rescaling,9=neighbhour of eyes
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(w+x,h+y),(0,0,255),2)
		cv2.rectangle(frame,(x,y-30),(x+h,y),(0,0,0), thickness=cv2.FILLED)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame,b2,(x+15,y), font, 1, (255,255,255), 2, cv2.LINE_AA)
	cv2.namedWindow("my_image",cv2.WINDOW_NORMAL)
	cv2.imshow("my_image",frame)

	cv2.resizeWindow("my_image",640,480)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		x=b2
		break
camera.release()
Instance = vlc.Instance()
player = Instance.media_player_new()
a=["/home/ankurgupta9352/Downloads/jhanjar.mp4","/home/ankurgupta9352/Downloads/life.mp4"]
b=["/home/ankurgupta9352/Downloads/dooriyan.mp4","/home/ankurgupta9352/Downloads/tera-ghata.mp4"]
c=["/home/ankurgupta9352/Downloads/kar har.mp4","/home/ankurgupta9352/Downloads/prada.mp4"]
if x=="happy" or x=="neutral":
	subprocess.Popen(['vlc', '-vvv',random.choice(a)])
if x=="sad" or x=="anger":
	subprocess.Popen(['vlc', '-vvv',random.choice(b)])
if x=="surprise"or x == "fear":
	subprocess.Popen(['vlc', '-vvv',random.choice(c)])
cv2.destroyAllWindows()

	#print(img)
# #0=anger,discused,fear,happy,sad,surprise,neutral