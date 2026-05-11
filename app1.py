from flask import Flask,redirect, url_for,render_template,request
import os
from index import d_dtcn
import sys

secret_key = str(os.urandom(24))

python_exe = sys.executable

app = Flask(__name__)
app.config['TESTING'] = True
app.config['DEBUG'] = True
app.config['FLASK_ENV'] = 'development'
app.config['SECRET_KEY'] = secret_key
app.config['DEBUG'] = True

# Defining the home page of our site
@app.route("/",methods=['GET', 'POST'])
def home():
    print(request.method)
    if request.method == 'POST':
        if request.form.get('Continue') == 'Continue':
           return render_template("test1.html")
    else:
        # pass # unknown
        return render_template("index.html")
    
@app.route("/start", methods=['GET', 'POST'])
def start_detection():
    if request.method == 'POST':
        # Run Web Cam script
        if request.form.get('StartWebCam') == 'StartWebCam':
            os.system(f'"{python_exe}" drowsiness_detection.py --shape_predictor shape_predictor_68_face_landmarks.dat')
        # Run Phone Cam script
        elif request.form.get('StartPhoneCam') == 'StartPhoneCam':
            os.system(f'"{python_exe}" android_cam.py --shape_predictor shape_predictor_68_face_landmarks.dat')
    return render_template("start.html")

@app.route('/contact', methods=['GET', 'POST'])
def cool_form():
    if request.method == 'POST':
        # do stuff when the form is submitted
        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('index'))

    # show the form, it wasn't submitted
    return render_template('contact.html')

if __name__ == "__main__":
    app.run()