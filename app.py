import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, flash, redirect, render_template, request, url_for, send_from_directory,abort
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

mnist_model = load_model('model/mnist.h5')

app = Flask(__name__)

app.secret_key = 'vatsalparsaniya'

app.config["MNIST_BAR"] = "generated_image"
app.config["IMAGES"] = "upload"

@app.route('/')
def home():
    flash("Hello Welcome to Vatsal's ML-DL Model Deploy site")
    return render_template('index.html')

@app.route('/mnist/')
def mnist_home():
    return render_template('mnist.html')

@app.route('/mnistprediction/', methods=['GET', 'POST'])
def mnist_prediction():
    if request.method == "POST":
        if not request.files['file'].filename:
            flash("No File Found")
        else:
            f =  request.files['file']
            f.save("uploads/"+f.filename)
            tf_image = image.load_img("uploads/"+f.filename, 
                            grayscale=True, 
                            color_mode='rgb', 
                            target_size=(28,28),
                            interpolation='nearest'
                            )
            np_image = image.img_to_array(tf_image)
            
            pred_img = np.reshape(np_image,(1,28,28,1))/255.0
            pred_img = 1 - pred_img
            predictions = mnist_model.predict(pred_img)
            number = int(np.argmax(predictions))
            print(number)

            plt.figure()
            y_pos = np.arange(10)
            plt.bar(y_pos, predictions[0])
            plt.savefig('generated_image/'+f.filename)

            return str(number)

@app.route("/get-mnist-image/<image_name>")
def get_mnist_image(image_name):
    try:
        return send_from_directory(app.config["MNIST_BAR"], filename=image_name)
    except FileNotFoundError:
        abort(404)


if __name__ == '__main__':
    app.run(threaded=True)

