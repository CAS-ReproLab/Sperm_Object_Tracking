from flask import Flask, render_template, request


app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=["POST"])
def process():
    slider_value = request.form["slider"]
    
    templateData ={
        'slider_value': slider_value
    }
    
    return render_template('index.html', **templateData)

if __name__ == '__main__':
    app.run(host='127.0.0.1')