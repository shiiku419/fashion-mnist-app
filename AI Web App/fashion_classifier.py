import os, shutil
from flask import Flask, request, redirect, url_for, render_template, Markup
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from PIL import Image
import numpy as np

UPLOAD_FOLDER = "./static/images/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
n_class = len(labels)
img_size = 28
n_result = 3  

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/result", methods=["GET","POST"])
def result():
    if request.method == "POST":
      
        if "file" not in request.files:
            print("File doesn't exist!")
            return redirect(url_for("index"))
        file = request.files["file"]
        if not allowed_file(file.filename):
            print(file.filename + ": File not allowed!")
            return redirect(url_for("index"))

       
        if os.path.isdir(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER) 
        os.mkdir(UPLOAD_FOLDER)
        filename = secure_filename(file.filename)  
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        
        image = Image.open(filepath)
        image = image.convert("RGB")
        image = image.resize((img_size, img_size))
        x = np.array(image, dtype=float)
        x = x.reshape(3,img_size, img_size) / 255        

        
        model = load_model("fashion.h5")
        y = model.predict(x)[0]
        sorted_idx = np.argsort(y)[::-1]  
        result = ""
        for i in range(n_result):
            idx = sorted_idx[i]
            ratio = y[idx]
            label = labels[idx]
            result += "<p>" + str(round(ratio*100, 1)) + "%の確率で" + label + "です。</p>"
        return render_template("result.html", result=Markup(result), filepath=filepath)
    else:
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)