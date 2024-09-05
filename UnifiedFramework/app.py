from flask import Flask, render_template, request, Response, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import time

class ConfigInfo:
    def __init__(self):
        self.current_filename = "demo.mp4"
        self.use_median = False
        self.diameter = 11
        self.minmass = 100
        self.fps = 30
        self.width = 640
        self.height = 480
        self.first_frame = None
        
    def set_video_info(self, video):
        self.fps = video.get(cv2.CAP_PROP_FPS)
        self.width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.first_frame = video.read()[1]
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)


config_info = ConfigInfo()

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
    global config_info
    video = cv2.VideoCapture("cache/" + config_info.current_filename)
    saveVideo(video, "cache/output.mp4")

    return render_template('index.html')

@app.route('/success', methods = ['POST'])   
def success():   

    global config_info

    if request.method == 'POST':

        file = request.files["video file"]
        filename = secure_filename(file.filename)

        file.save("cache/" + filename)
        config_info.current_filename = filename

        video = cv2.VideoCapture("cache/" + filename)
        config_info.set_video_info(video)
        saveVideo(video, "cache/output.mp4")

        return render_template('acknowledgement.html')

@app.route('/uploads/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cache_image')
def cache_image():
    return send_from_directory(app.config['UPLOAD_FOLDER'], "output.jpg")    

@app.route('/preprocess', methods=["POST"])
def process():

    global config_info
    video = cv2.VideoCapture("cache/" + config_info.current_filename)
    out = cv2.VideoWriter("cache/output.mp4", cv2.VideoWriter_fourcc(*'avc1'), config_info.fps, (2*config_info.width, config_info.height))

    use_median = request.form.get("use_median")
    print("USE_MEDIAN =", use_median)

    for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = video.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if use_median == "on":
            config_info.use_median = True
            gray = gray.astype(np.float32)
            gray = np.abs(gray - np.median(gray))
            gray = np.clip(gray,0,255).astype(np.uint8)
        else:
            config_info.use_median = False

        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        out.write(np.hstack([frame, result]))

    out.release()

    templateData ={
        'use_median': use_median
    }

    return render_template('index.html', **templateData)


@app.route('/detect', methods=["GET","POST"])
def detect():

    global config_info

    image = config_info.first_frame

    if request.method == 'GET':
        cv2.imwrite("cache/output.jpg", image)
        return render_template('detection.html')


    diameter = request.form.get("diameter")
    minmass = request.form.get("minmass")
    print("DIAMETER =", diameter)
    print("MINMASS =", minmass)

    cv2.imwrite("cache/output.jpg", image)
    
    templateData ={
        'diameter': diameter
    }

    return render_template('detection.html', **templateData)

if __name__ == '__main__':
    app.run(host='127.0.0.1',debug=True)