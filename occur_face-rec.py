from asyncore import write
import numpy as np
# import argparse
# import imutils
import time
import dlib
import cv2
# import uuid 
import os
import math
# import glob 
import face_recognition
import pickle
from collections import deque
# import random
import pandas as pd

import mediapipe as mp
import requests


# from scipy.spatial import distance as dist
# from imutils.video import FileVideoStream
# from imutils.video import VideoStream
# from imutils import face_utils
# from imutils.video import FPS
# from threading import Timer


# cap = cv2.VideoCapture(0)

def twoArgs(arg1):
    arg1.release()

def gen_capture(stream = None, url = 0,fps=None):
    # time.sleep(1.0)
    if(fps!=None):
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    if stream is None:
        stream = cv2.VideoCapture(url)
        stream.set(cv2.CAP_PROP_BUFFERSIZE,1)
        return stream
    else:
        stream.release()
        stream = cv2.VideoCapture(url)
        stream.set(cv2.CAP_PROP_BUFFERSIZE,1)
        return stream
    

def take_known_picture(url = 0):
    # url = "rtsp://admin:admin@192.168.20.36:8080"
    # cap = cv2.VideoCapture(0)
    name = str(input("enter name of the picture: "))
    known_dir="known_images/"+name
    cap = gen_capture(url=0)
    # ret,frame = cap.read()
    # cv2.imshow("name",frame)
    if not (os.path.isdir(known_dir)):
        mode = 0o777
        os.makedirs(known_dir,mode)
    while True:
        ret,frame = cap.read()
        if not ret:
            time.sleep(0.1)
            cap = gen_capture(stream=cap)
        cv2.imshow("name",frame) #display the captured image
        if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y' 
            file_path = known_dir+'/01.jpg'
            cv2.imwrite(file_path,frame)
            cv2.destroyAllWindows()
            break
    cap.release()
    return name


def midpoint(point1 ,point2):
    return (point1.x + point2.x)/2,(point1.y + point2.y)/2

def euclidean_distance(point1 , point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_blink_ratio(eye_points, facial_landmarks):
    
    #loading all the required points
    corner_left  = (facial_landmarks.part(eye_points[0]).x, 
                    facial_landmarks.part(eye_points[0]).y)
    corner_right = (facial_landmarks.part(eye_points[3]).x, 
                    facial_landmarks.part(eye_points[3]).y)
    
    center_top    = midpoint(facial_landmarks.part(eye_points[1]), 
                             facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), 
                             facial_landmarks.part(eye_points[4]))

    #calculating distance
    horizontal_length = euclidean_distance(corner_left,corner_right)
    vertical_length = euclidean_distance(center_top,center_bottom)

    ratio = horizontal_length / vertical_length
    # print("ratio: " + str(ratio))
    return ratio, horizontal_length




def blinking_count(expected_blinks,url = 0):
    blinks = 0
    # BLINK_RATIO_THRESHOLD = 5.2
    end_time = time.time()+35
    result = False
    time_queue = deque([0]*3,maxlen=3)
    # url = 'rtsp://192.168.1.60:554/user=admin&password=&channel=1&stream=1.sdp?'
    cap = gen_capture(url=url)

    while True:
        # Capture the image from the webcam
        ret, image = cap.read()
        if not ret:
            time.sleep(1.0)
            cap = gen_capture(stream=cap,url=url)
            ret, image = cap.read()

        # Convert the image color to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect the face
        # rects = detector(gray, 1)
        # Detect landmarks for each face
        faces,_,_ = detector.run(image = image, upsample_num_times = 0, 
                           adjust_threshold = 0.0)
        rects=faces
        for rect in rects:
            # Get the landmark points
            shape = predictor(gray, rect)
            left_eye_ratio ,left_length = get_blink_ratio(left_eye_landmarks, shape)
            right_eye_ratio,right_length = get_blink_ratio(right_eye_landmarks, shape)
            blink_ratio     = (left_eye_ratio + right_eye_ratio) / 2
            avg_len = (left_length+right_length)/2
        # Convert it to the NumPy Array
            shape_np = np.zeros((68, 2), dtype="int")

            for i in range(0, 68):
                shape_np[i] = (shape.part(i).x, shape.part(i).y)
            shape = shape_np

            # Display the landmarks
            for i, (x, y) in enumerate(shape):
            # Draw the circle to mark the keypoint 
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

            # text = "BLINK_COUNT: "+str(blinks)
            if avg_len>30.1:
                BLINK_RATIO_THRESHOLD = avg_len/6.1
            elif avg_len<=30.1 and avg_len>=20.1:
                BLINK_RATIO_THRESHOLD = avg_len/5.05
            else:
                BLINK_RATIO_THRESHOLD = avg_len/2.59
            
            if blink_ratio > BLINK_RATIO_THRESHOLD:
                #Blink detected! Do Something!

                blinks = blinks+1
                time.sleep(.195)
                time_queue.append(round(time.time()*1000))
                if not (time_queue[0]== 0 or time_queue[2] == 0):
                    if (time_queue[2]-time_queue[0]<1000):
                        # print("bot_time :"+str(time_queue[2]-time_queue[0]))
                        blinks = 0
            text = "BLINK_COUNT: "+str(blinks)
            cv2.putText(image,text,(10,50), cv2.FONT_HERSHEY_SIMPLEX,
                        2,(0,0,0),2,cv2.LINE_AA)
            cv2.putText(image,str(blink_ratio),(60,100), cv2.FONT_HERSHEY_SIMPLEX,
                            2,(0,0,0),2,cv2.LINE_AA)
            cv2.putText(image,str(avg_len),(50,150), cv2.FONT_HERSHEY_SIMPLEX,
                            2,(0,0,0),2,cv2.LINE_AA)
            if (blinks >=expected_blinks):
               result = True
               cv2.destroyAllWindows()
               return result

        # Display the image
        ims = cv2.resize(image,(1920,1080))
        cv2.imshow('BlinkDetector', ims)

        if (time.time() > end_time):
            print("please follow the instructions")
            cv2.destroyAllWindows()
            break
        # Press the escape button to terminate the code
        if cv2.waitKey(10) == 27:
            cv2.destroyAllWindows()
            break

    cap.release()
    return result

known_face_encodings=[]
known_face_names=[]
if not os.path.exists("encodings.pkl"):
    my_list = os.listdir('known_images')
    for i in range(len(my_list)):
        if(my_list[i]!=".ipynb_checkpoints"):

            image=face_recognition.load_image_file("known_images/"+my_list[i]+"/02.jpg")
            print(my_list[i])
            face_encoding = face_recognition.face_encodings(image,num_jitters=100)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(my_list[i])

    with open('encodings.pkl','wb') as f:
        pickle.dump([known_face_encodings,known_face_names], f)
else:
    with open('encodings.pkl', 'rb') as f:
        known_face_encodings ,known_face_names = pickle.load(f)
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()
distances = []

def face_recognize(url = 0):
    testDf = pd.DataFrame(columns=known_face_names)
    # video_capture = cv2.VideoCapture(0)
    video_capture = gen_capture(url=url)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 200.0, (int(video_capture.get(3)), int(video_capture.get(4))))
    # srart_time = time.time()
    # end_time = time.time()+15
    # fps = FPS().start()
    tTime = 0.0
    pTime = 0
    pName = []
    while True:
        # video_capture.set(cv2.CAP_PROP_POS_FRAMES,0)
        # if end_time<= time.time():
        #     print("destroying")
        #     # end_time = time.time()+5
        #     video_capture.release()
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if frame is not None:
            frame = cv2.resize(frame, (640,480))
            # frame = cv2.resize(frame, (800,600))
            # frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)

        #time out
        if not ret:
            # time.sleep(1.0)
            video_capture = gen_capture(stream=video_capture,url=url)
            ret, frame = video_capture.read()
            frame = cv2.resize(frame, (640,480))
            # frame = cv2.resize(frame, (800,600))
            # frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = frame
        results = faceDetection.process(rgb_frame)
        # cv2.imshow('FaceDetector', rgb_frame)
        face_locations = []
        if results.detections:
            if tTime == 0.0:
                tTime = time.time()
                print("start time : "+str(tTime))
            for id,detection in enumerate(results.detections):
                # mpDraw.draw_detection(rgb_frame, detection)
                bBoxC=detection.location_data.relative_bounding_box
                ih,iw,ic=rgb_frame.shape
                bBox = int(bBoxC.xmin*iw),int(bBoxC.ymin*ih),int(bBoxC.width*iw),int(bBoxC.height*ih)
                left,top,right,bottom = bBox[1],bBox[0]+bBox[2],bBox[1]+bBox[3],bBox[0]
                # cv2.rectangle(rgb_frame,(left,top),(right,bottom),(0,255,0),2)
                # print(left,top,right,bottom )
                tup=(left,top,right,bottom)
                face_locations.append(tup)


        # print(face_locations)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(rgb_frame, "FPS: {:.2f}".format(fps), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Find all the faces and face enqcodings in the frame of video
        # face_locations = face_recognition.face_locations(rgb_frame,number_of_times_to_upsample=1)
        #print(face_locations)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        #print("line")
        #print(face_encodings)
        # Loop through each face in this frame of video
        faces = []
        count = 0
        dTime = time.time()
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            # count+=1
            eTime = time.time()
            if(eTime-dTime>=1.0):
                pName=[]
                print("time out array clearence")
                dTime = time.time()
            print(face_locations)
            name = "Unknown"
            #single face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] < 0.6:
                name1 = known_face_names[best_match_index]
                # if count >= 50:
                #     count = 0
                #     pName=[]
                if len(pName)>25:
                    print("array clearence")
                    for _ in range(25): pName.pop(0)
                pName.append(name1)
            
            if(len(pName)>10):
                name = max(pName, key=pName.count)
                print(len(pName))
                print(pName)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            
            if (time.time()-dTime)==1.5:
                post_url = "http://192.168.10.72"
                r = requests.post(post_url,  data={'name': name, 'time': str(time.time())})
                my_file = open('namefile.txt','w')
                my_file.write(name)

            #file=random.getrandbits(32)
            #cv2.imwrite('./images/'+str(file)+'.png',frame)
            vid = cv2.flip(frame,1)
            out.write(vid)

        cv2.imshow('FaceDetector', rgb_frame)
        key=cv2.waitKey(1)
    #     # Hit 'q' on the keyboard to quit!
        if key%256 == 27:
            cv2.destroyAllWindows()
            # testDf = pd.DataFrame(distances , columns=known_face_names)
            # testDf.to_csv('testDf.csv')
            # print(distances)
            break

    # # Release handle to the webcam

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

# url = "rtsp://192.168.1.60:554/user=admin&password=&channel=1&stream=1.sdp?"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
left_eye_landmarks  = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]
#url = 0
# name = take_known_picture(url=0)
blink_num = 4
# res = blinking_count(blink_num,url)
res = True

#url = "rtsp://admin:PE-LD-04@192.168.10.33:554/media/video2"
url = "rtsp://admin:admin321!!@192.168.10.33:554/ch01/0"
# url = 0
# url="vid1.webm"
#url="pexels-cottonbro-5329613.mp4"
if res:
    face_recognize(url)
    print(distances)
    testDf = pd.DataFrame(distances , columns=known_face_names)
    testDf.to_csv('testDf.csv')

else:
    print("please follow the instructions")
