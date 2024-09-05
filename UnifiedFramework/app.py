from flask import Flask, render_template, request, Response, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import time

current_filename = "demo.mp4"

UPLOAD_FOLDER = 'cache'
ALLOWED_EXTENSIONS = {'avi', 'mp4'}

def saveVideo(video, file_name):
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
    
    for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = video.read()
        out.write(frame)

    out.release()

def video_stream():
    video = cv2.VideoCapture("cache/output.mp4")
    fps = video.get(cv2.CAP_PROP_FPS)
    while video.isOpened():
        time.sleep(1/(fps+1e-4))
        ret, frame = video.read()
        if not ret:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    global current_filename
    video = cv2.VideoCapture("cache/" + current_filename)
    saveVideo(video, "cache/output.mp4")

    return render_template('index.html')

@app.route('/success', methods = ['POST'])   
def success():   

    global current_filename

    if request.method == 'POST':

        file = request.files["video file"]
        filename = secure_filename(file.filename)

        file.save("cache/" + filename)
        current_filename = filename

        video = cv2.VideoCapture("cache/" + filename)
        saveVideo(video, "cache/output.mp4")

        return render_template('acknowledgement.html')

@app.route('/uploads/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/preprocess', methods=["POST"])
def process():

    global current_filename

    video = cv2.VideoCapture("cache/" + current_filename)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("cache/output.mp4", cv2.VideoWriter_fourcc(*'avc1'), fps, (2*width, height))

    use_median = request.form.get("use_median")
    print("USE_MEDIAN =", use_median)

    for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = video.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if use_median == "on":
            gray = gray.astype(np.float32)
            gray = np.abs(gray - np.median(gray))
            gray = np.clip(gray,0,255).astype(np.uint8)

        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        out.write(np.hstack([frame, result]))

    out.release()

    templateData ={
        'use_median': use_median
    }

    return render_template('index.html', **templateData)

if __name__ == '__main__':
    app.run(host='127.0.0.1',debug=True)