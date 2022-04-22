import flask
from flask import Flask,render_template,request
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = "img"

@app.route('/', methods=['GET'])
def hello_world():
    return render_template("index.html")



@app.route('/predict', methods= ['POST'])
def predict():
    IMG_SIZE = 128
    imgfile=request.files['imagefile']
    imgfile.save(os.path.join(app.config['UPLOAD_FOLDER'], imgfile.filename))
    image_path="./img/"+ imgfile.filename
    img = cv2.resize(cv2.imread(image_path),(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_AREA)
    model1 = keras.models.load_model("VGG_model3_128_20ep.h5")
    img = img.reshape(-1,IMG_SIZE,IMG_SIZE,3)
    img = img/255
    img_final = img.reshape(-1,128,128,3)
    preds   = model1.predict(img_final)
    res = np.argmax(preds, axis=1)
    return render_template("index.html", pred = res)



if __name__ == '__main__':
    app.run(debug=True)