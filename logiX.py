import win32com.client as wincl
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import xlwt
from xlwt import Workbook
import urllib
import json
import random
from threading import Thread
from firebase import firebase
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn import svm
df=pd.read_csv('H:/enginx/pycodes/foofin.csv');
col=['Speed','Bphm']
X=df[col]
y=df['Out']
clf= LogisticRegression()
clf1= RandomForestClassifier(max_depth=2,random_state=0)
clf2=KNeighborsClassifier(n_neighbors=3)
clf3 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(6,2), random_state=1)
clf4 = svm.SVC()
clf.fit(X,y)
clf1.fit(X,y)
clf2.fit(X,y)
clf3.fit(X,y)
clf4.fit(X,y) 
import os
import thingspeak
import urllib
import json
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn as sns
from sklearn import svm

COUNTER = 0
SND=0
SND1=0
TOTAL = 0
TOTAL1 = 0
TOTAL2=0
SPEED=0
i=0
j=0
j1=0
def convert(x):
    a=x.split(':')
    x=a[5]
    x=x[1:x[1:].find('"')+1]
    x=int(x)
    return x
 
def send_mail(recipient, subject, message):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    username = "logixpltd@gmail.com"
    password = "timebomb321"

    msg = MIMEMultipart()
    msg['From'] = username
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(message))

    print('sending mail to ' + recipient + ' on ' + subject)
    mailServer = smtplib.SMTP('smtp.gmail.com', 587)
    mailServer.ehlo()
    mailServer.starttls()
    mailServer.ehlo()
    mailServer.login(username, password)
    mailServer.sendmail(username, recipient, msg.as_string())
    mailServer.close()
    
def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
def f1():
    wb=Workbook()
    s1 = wb.add_sheet('Sheet 1')
    global COUNTER,SND,SND1,TOTAL,TOTAL1,TOTAL2,SPEED,i,j,j1,x
    camera = cv2.VideoCapture(0)
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] 
    while True:
    	
        if(i%66==0):
            SND1=TOTAL-TOTAL2
            s1.write(j,0,SND1)
            TOTAL2=TOTAL
            j=j+1
        if(i%100==0):
            if j1>120:
                j1=0;
            SPEED=random.randint(j1,j1+20);
            j1=j1+20
        ret,frame = camera.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray,rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                COUNTER = 0
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            wb.save('exceln.xls')
            break
        i=i+1
   
    cv2.destroyAllWindows()
    camera.stop()
    
def f2():
    global TOTAL,TOTAL1,SPEED
    fr=firebase.FirebaseApplication('https://a124-270e9.firebaseio.com/')
    while True:
        time.sleep(25)
        TS1 = urllib.request.urlopen("https://api.thingspeak.com/update?api_key=EGB3IM8MPQH4U8VK&field6="+str(SPEED)+"&field3="+str(TOTAL-TOTAL1))
        TS1.close()
        result1=fr.put('/','Blinkrate',str(TOTAL-TOTAL1))
        result2=fr.put('/','Speed',str(SPEED))
        TOTAL1=TOTAL
def f3():
    time.sleep(37)
    p=0#Temperature
    p1=0#Blink_Warning
    p2=0#Blink_Danger
    p3=0#Humidity
    p4=0#Flame
    p5=0#Vibration
    p6=0#Collision
    READ_API_KEY_1='4W5UVHGWABFCNU0E'
    CHANNEL_ID_1= '562330'
    speak = wincl.Dispatch("SAPI.SpVoice")
    while True:
        h0=0
        h1=0
        h2=0
        ch = thingspeak.Channel(id=CHANNEL_ID_1,api_key=READ_API_KEY_1)
        x=ch.get_field_last(1)
        a1=convert(x)#Temperature
        x=ch.get_field_last(3)
        a3=convert(x)#Blink rate
        x=ch.get_field_last(6)
        a6=convert(x)#Speed
        x=ch.get_field_last(2)
        a2=convert(x)#Humidity
        x=ch.get_field_last(7)
        a7=convert(x)#Flame
        x=ch.get_field_last(5)
        a5=convert(x)#Vibration
        x=ch.get_field_last(4)
        a4=convert(x)#Collision
        if(a1>29 and p==0):
            send_mail('yesh385@gmail.com','Warning!','Temperature of package compromised')
            speak.Speak("Temperature Compromised")
            p=1
        if(a2>60 and p3==0):
            send_mail('yesh385@gmail.com','Warning!','Humidity of package compromised')
            speak.Speak("Humidity Compromised")
            p3=1
        if(a7==1 and p4==0):
            send_mail('yesh385@gmail.com','Warning!','Flame Alert')
            speak.Speak("Flame Alert")
            p4=1
        if(a5>18000 and p5==0):
            send_mail('yesh385@gmail.com','Warning!','Vibration Alert')
            speak.Speak("Vibration Alert")
            p5=1
        Reaction_dist=a4*0.3
        Braking_dist=((a4/10)**2)*0.4
        dist=Reaction_dist+Braking_dist+100
        if(dist>=a4 and p6==0):
            send_mail('yesh385@gmail.com','Warning!','Slow Down')
            speak.Speak("Slow Down")
            p6=1
        if(clf.predict([[a6,a3]])==2):
            h2=h2+1
        elif(clf.predict([[a6,a3]])==1):
            h1=h1+1
        elif(clf.predict([[a6,a3]])==0):
            h0=h0+1
        if(clf1.predict([[a6,a3]])==2):
            h2=h2+1
        elif(clf1.predict([[a6,a3]])==1):
            h1=h1+1
        elif(clf1.predict([[a6,a3]])==0):
            h0=h0+1
        if(clf2.predict([[a6,a3]])==2):
            h2=h2+1
        elif(clf2.predict([[a6,a3]])==1):
            h1=h1+1
        elif(clf2.predict([[a6,a3]])==0):
            h0=h0+1
        if(clf3.predict([[a6,a3]])==2):
            h2=h2+1
        elif(clf3.predict([[a6,a3]])==1):
            h1=h1+1
        elif(clf3.predict([[a6,a3]])==0):
            h0=h0+1
        if(clf4.predict([[a6,a3]])==2):
            h2=h2+1
        elif(clf4.predict([[a6,a3]])==1):
            h1=h1+1
        elif(clf4.predict([[a6,a3]])==0):
            h0=h0+1
        if(max(h0,h1,h2)==h1 and p1==0):
            send_mail('yesh385@gmail.com','Warning!\n','Accident Warning')
            p1=1
            for e in range(5):
                speak.Speak("Accident Warning")
        elif(max(h0,h1,h2)==h2 and p2==0):
            send_mail('yesh385@gmail.com','Warning!\n','Accident Danger')
            p2=1
            for e in range(5):
                speak.Speak("Accident Danger")
        
            
if __name__ == '__main__':
    Thread(target = f1).start()
    Thread(target = f2).start()
    Thread(target = f3).start()