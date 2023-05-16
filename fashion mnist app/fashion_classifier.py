import os
import shutil
from flask import Flask, request, redirect, url_for, render_template, Markup
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from PIL import Image, ImageOps
import numpy as np

print(os.getcwd())

UPLOAD_FOLDER = "/Users/shiikushota/fashion-mnist-app/fashion mnist app/static/test"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

n_class = len(labels)
img_size = 28
n_result = 1

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


model = load_model(
    "/Users/shiikushota/fashion-mnist-app/fashion mnist app/fashion.h5")
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/result", methods=["GET", "POST"])
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
        image = image.convert('RGB')
        image = ImageOps.invert(image)
        image = image.resize((img_size, img_size))
        image = np.array(image, dtype=float)
        image = 0.299 * image[:, :, 0] + 0.587 * \
            image[:, :, 1] + 0.114 * image[:, :, 2]
        image = image.reshape(1, img_size, img_size, 1) / 255

        y = model.predict(image)[0]
        print(y)
        print(image.shape)
        predicted = y.argmax()
        result = ""
        label = labels[predicted]
        result += "<p>" + "画像は" + label + "です。</p>"
        return render_template("result.html", result=Markup(result), filepath=filepath)


if __name__ == "__main__":
    app.run(debug=True)
