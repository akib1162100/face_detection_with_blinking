# import the necessary packages

from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
from camera import VideoCamera

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

known_face_encodings=[]
known_face_names=[]

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

def detect_motion(frameCount):

    global vs, outputFrame, lock
    url = "rtsp://admin:admin321!!@192.168.10.33:554/ch01/0"
    video_capture = cv2.VideoCapture(url)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE,1)
    while True:
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

        
        
        with lock:
            outputFrame = frame.copy()
	





def gen(camera):
    n=0
    while True:
        
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    resp = Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
            

# check to see if this is the main thread of execution
if __name__ == '__main__':
	
	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=32)
	t.daemon = True
	t.start()
	# start the flask app
	app.run(host='192.168.10.89', debug=False,
		threaded=True, use_reloader=False)
# release the video stream pointer
vs.stop()