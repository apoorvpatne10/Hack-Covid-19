# usage: python app.py
import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
import argparse
import cv2
import time
import uuid
import base64

from fastai.vision import *

tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
path = Path('')

data = ImageList.from_csv(path, 'training.csv', cols=0, folder='covid19-dataset/images', suffix='')
data = data.split_by_rand_pct(0.1)\
       .label_from_df(cols=1)\
       .transform(get_transforms(), size=224, resize_method=3)\
       .databunch(bs=32)\
       .normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet101, metrics=[error_rate,accuracy]).load("stage-2")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def predict(file):
	img = open_image(file)
	img = img.apply_tfms(tfms=get_transforms()[1], size=224, resize_method=3)
	res = learn.predict(img)
	result = []
	result.append(str(res[0]))
	result.append(float(res[2][int(res[1])]))

	return result

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='../uploads/icon.jpeg')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = predict(file_path)
            #result = 18

            label=result[0]
            res = str(result[1])
            label = label+"{ Score : "+res+"}"
            print(result)
            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))
            return render_template('template.html', label=label, imagesource='../uploads/' + filename)

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=True
    app.run(host='0.0.0.0', port=3000)
