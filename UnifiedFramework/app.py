from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import cv2
import numpy as np

current_filename = "demo.mp4"

def saveVideo(video, file_name):
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
    
    for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = video.read()
        out.write(frame)

    out.release()


app = Flask(__name__, static_folder='static')


@app.route('/')
def index():
    global current_filename
    video = cv2.VideoCapture("static/" + current_filename)
    saveVideo(video, "static/output.mp4")

    return render_template('index.html')

@app.route('/success', methods = ['POST'])   
def success():   

    global current_filename

    if request.method == 'POST':

        file = request.files["video file"]
        filename = secure_filename(file.filename)

        file.save("static/" + filename)
        current_filename = filename

        video = cv2.VideoCapture("static/" + filename)
        saveVideo(video, "static/output.mp4")

        return render_template('acknowledgement.html')


@app.route('/process', methods=["POST"])
def process():

    global current_filename

    video = cv2.VideoCapture("static/" + current_filename)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("static/output.mp4", cv2.VideoWriter_fourcc(*'avc1'), fps, (2*width, height))

    slider_value = request.form["threshold"]

    for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = video.read()
        
        threshold = int(slider_value)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        out.write(np.hstack([frame, binary]))

    out.release()

    templateData ={
        'threshold': slider_value
    }

    return render_template('index.html', **templateData)

if __name__ == '__main__':
    app.run(host='127.0.0.1',debug=True)