import cv2
import numpy
from flask import Flask, render_template, Response, stream_with_context, Request

video_fn = "ExampleVideos/10X_-ph_9fps_R1.avi"

video = cv2.VideoCapture(video_fn)
app = Flask(__name__)

def video_stream():
    while(True):
        ret, frame = video.read()
        if not ret:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        else:
            ret, buffer = cv2.imencode('.jpeg',frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/test')

def test():
    return render_template('test.html')

@app.route('/video_feed')

def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host ='127.0.0.1', port= '5000', debug=False)