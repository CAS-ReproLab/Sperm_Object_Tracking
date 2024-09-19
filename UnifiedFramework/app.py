from flask import Flask, render_template, request, Response, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import trackpy as tp
import time
import utils
import tracker
import visualizer

class ConfigInfo:
    def __init__(self):
        self.current_filename = "demo.mp4"
        self.use_median = False
        self.diameter = 11
        self.minmass = 300
        self.search_range = 7
        self.memory = 3
        self.fps = 5
        self.width = 1024
        self.height = 1024
        self.first_frame = cv2.imread("cache/demo.jpg")
        self.detect_df = None
        self.track_df = None
        
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

    else:
        diameter = int(request.form.get("diameter"))
        minmass = int(request.form.get("minmass"))
        print("DIAMETER =", diameter)
        print("MINMASS =", minmass)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        df = tp.locate(gray, diameter=diameter, minmass=minmass)

        output_im = np.copy(image)

        for row in df.iterrows():
            x = int(row[1]['x'])
            y = int(row[1]['y'])
            output_im = cv2.circle(output_im, (x,y), 3, (0, 0, 255), -1)

        cv2.imwrite("cache/output.jpg", np.hstack([image, output_im]))
        
        config_info.diameter = diameter
        config_info.minmass = minmass


    templateData ={
        'diameter': config_info.diameter,
        'minmass': config_info.minmass
    }

    return render_template('detection.html', **templateData)

@app.route('/track', methods=["GET","POST"])
def track():

    if request.method == 'GET':
        diameter = config_info.diameter
        minmass = config_info.minmass

        frames = utils.loadVideo("cache/" + config_info.current_filename,as_gray=True)
        df = tracker.determineCentroids(frames, diameter=diameter, minmass=minmass)

        config_info.detect_df = df

    else:
        search_range = int(request.form.get("search_range"))
        memory = int(request.form.get("memory"))
        print("SEARCH_RANGE =", search_range)
        print("MEMORY =", memory)

        df = tracker.trackCentroids(config_info.detect_df, search_range=config_info.search_range, memory=config_info.memory)
        config_info.track_df = df

        # Save the video with the tracking
        video = cv2.VideoCapture("cache/" + config_info.current_filename)
        out = cv2.VideoWriter("cache/output.mp4", cv2.VideoWriter_fourcc(*'avc1'), config_info.fps, (2*config_info.width, config_info.height))
        
        max_id = int(df['sperm'].max())
        colors = np.random.randint(0, 255, (max_id+1, 3))
        ret, frame = video.read()
        mask = np.zeros_like(frame)

        for frame_num in range(1,int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = video.read()
            img = visualizer.opticalFlow(frame,df,frame_num,mask,colors)
            out.write(np.hstack([frame, img]))

        config_info.search_range = search_range
        config_info.memory = memory

    templateData = {
        'search_range': config_info.search_range,
        'memory': config_info.memory
    }

    return render_template('track.html',**templateData)

if __name__ == '__main__':
    app.run(host='127.0.0.1',debug=True)