#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
hand_cascade=cv2.CascadeClassifier("yess.xml")
ds_factor=0.6
tracker = cv2.TrackerCSRT_create()
x=0
y=0
w=0
h=0

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        success, image = self.video.read()
        
        #image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
       # gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
       # face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        bbox=hand_cascade.detectMultiScale(image,1.1,5)
        global x
        global y
        global w
        global h
        x= bbox[0][0]
        y=bbox[0][1]
        w=bbox[0][2]
        h=bbox[0][3]
        hg=(x,y,w,h)
        tracker.init(image,hg)
       # for (x,y,w,h) in face_rects:
            
       # 	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
       # 	break
        #self.video.release()
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        #image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        #gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        success, bbox = tracker.update(image)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3, 3 )
        x1, y1, w1, h1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        #drawBox(frame,x1,y1,w1,h1)
        
        cv2.rectangle(image, (x1, y1), ((x1 + w1), (y1 + h1)), (255, 0, 255), 3, 3 )
        cv2.putText(image, "Tracking", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        #for (x,y,w,h) in face_rects:
        #
        #break
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

