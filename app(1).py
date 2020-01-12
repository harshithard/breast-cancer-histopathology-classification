import os
from flask import Flask, flash, render_template, redirect, request, url_for
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator,img_to_array
import cv2
from imutils import paths
import numpy as np
from flask import session

app = Flask(__name__)
app.config['SECRET_KEY'] = "supertopsecretprivatekey"
app.config['UPLOAD_FOLDER'] ="E:\gui\image"
pred=[]
conf=[]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        # show the upload form
        return render_template('homepage.html')

    if request.method == 'POST':
        # check if a file was passed into the POST request
        if 'image' not in request.files:
            flash('No file was uploaded.')
            return redirect(url_for('home'))

        image_file = request.files['image']

        # if filename is empty, then assume no upload
        if image_file.filename == '':
            flash('No file was uploaded.')
            return redirect(url_for('home'))

        # if the file is "legit"
        if image_file:
            passed = False
            try:
                filename = image_file.filename
                filepath = os.path.join('E:\gui\image', filename)
                image_file.save(filepath)
                passed = True
                model = load_model('E:\Downloads\mymodcat2.h5')
                imagepath = sorted(list(paths.list_images('E:\gui\image')))
                data = []
                for img in imagepath:
                    image=cv2.imread(img)
                    image = cv2.resize(image,(96,96))
                    image_array = img_to_array(image)
                    data.append(image_array)
                    data=np.array(data,dtype=float)
                    data = data/255.0
                    try:
                        prediction = model.predict(data)
                        pred.append(prediction)
                    except:
                        flash("environment problem!")
            except Exception:
                passed = False

            if passed:
                return redirect(url_for('predict'))
            else:
                flash('An error occurred, try again.')
                return redirect(url_for('home'))



@app.route('/prediction', methods=['GET', 'POST'])
def predict():  
    result=np.argmax(pred)
    #conf = max(pred)
    print(pred[0])
    if(result==0):
        temp="BENIGN"
    elif(result==1):
        temp="DUCTAL CARCINOMA"
    elif(result==2):
        temp="LOBULAR CARCINOMA"
    elif(result==3):
        temp="MUCINOUS CARCINOMA"
    elif(result==4):
        temp="PAPILLARY CARCINOMA"
    

    return render_template('predict.html', temp=temp ,conf=conf)




"""

@app.route('/aa/<filename>', methods=['GET'])
def images(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
ALLOWED_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpeg', 'gif'])

def is_allowed_file(filename):
    // Checks if a filename's extension is acceptable 
    allowed_ext = filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    return '.' in filename and allowed_ext

"""

if __name__ == "__main__":
    app.run(debug=True)