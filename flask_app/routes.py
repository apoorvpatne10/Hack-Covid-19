from werkzeug import SharedDataMiddleware
from flask import send_from_directory
import os
import cv2
import time
import json
import uuid
import base64
import requests
import argparse
import pandas as pd
import math
import re
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug import secure_filename
from geopy.distance import distance
from fastai.vision import *
from flask_app.models import Patient, Record
from flask_app.forms import PatientEntryForm, SuspectEntryForm
from flask_app import app, db
from bs4 import BeautifulSoup


# Fake news
URL = 'https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters'
page = requests.get(URL)

soup = BeautifulSoup(page.content, 'html.parser')
results = soup.find(id='PageContent_C002_Col01')

results = results.text.split("\n")
facts = []

for i in results:
	if(i != "" and i != "Download and share the graphic" and i!= " "):
		facts.append(i)


WORD = re.compile(r"\w+")

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

facts_vec = []
for i in facts:
	facts_vec.append(text_to_vector(i))

@app.route("/fake_news/")
def fk_news():
    return render_template("news.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    userText_vec = text_to_vector(userText)
    get_cosine_score = []

    for i in facts_vec:
    	get_cosine_score.append(get_cosine(i,userText_vec))

    return facts[get_cosine_score.index(max(get_cosine_score))]


########################################################

tfms = get_transforms(
    flip_vert=True,
    max_lighting=0.1,
    max_zoom=1.05,
    max_warp=0.)
path = Path('')

data = ImageList.from_csv(path, 'flask_app/covid19-dataset/training.csv', cols=0, folder='flask_app/covid19-dataset/images', suffix='')
data = data.split_by_rand_pct(0.1)\
       .label_from_df(cols=1)\
       .transform(get_transforms(), size=224, resize_method=3)\
       .databunch(bs=32)\
       .normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet101, metrics=[error_rate,accuracy]).load("stage-2")

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
    random = str(uuid.uuid4())  # Convert UUID format to a Python string.
    random = random.upper()  # Make all characters uppercase.
    random = random.replace("-", "")  # Remove the UUID '-'.
    return random[0:string_length]  # Return the random string.


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def allowed_data_files(filename):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1]
    if ext.upper() in app.config['ALLOWED_FILE_TYPES']:
        return True
    else:
        return False


def find_timeline_intersection(person_name, data, all_records):
    """
    Finds timeline intersections by comparing the suspect's timeline with existing
    records

    Args:
        suspect name (str): Person's name
        suspect data (json): Suspect's data
        all_records (db): Existing Records

    Returns:
        type: dataframe consisting of timeline intersections
    """

    result = pd.DataFrame(columns=['namea', 'nameb', 'place_a', 'place_b',
                                   'distance', 'delta (mins:secs)',
                                   'time_start_a', 'time_end_a'])
    c = 0
    for record in all_records:
        for activity in data['timelineObjects']:
            if 'placeVisit' not in activity.keys():
                continue

            if 'name' in activity['placeVisit']['location'].keys():
                place_name = activity['placeVisit']['location']['name']
            else:
                continue
            if 'address' in activity['placeVisit']['location'].keys():
                address = activity['placeVisit']['location']['address']
            else:
                continue

            start_long = round(float(activity['placeVisit']['location']['longitudeE7'])/1e7, 7)
            start_lat = round(float(activity['placeVisit']['location']['latitudeE7']) /1e7, 7)
            coord1 = (record.longitude, record.latitude)
            coord2 = (start_long, start_lat)

            dist = distance(coord1, coord2).m

            start1 = record.raw_time_start
            end1 = record.raw_time_end
            start2 = int(activity['placeVisit']['duration']
                         ['startTimestampMs']) / 1e3
            end2 = int(activity['placeVisit']['duration']
                       ['endTimestampMs']) / 1e3

            place_name1 = record.place_name
            place_name2 = activity['placeVisit']['location']['name']

            human_time_start = record.time_start
            human_time_end = record.time_end

            if dist <= 300 and (
                    start1 <= start2 <= end1 or start2 <= start1 <= end2):
                if start1 <= start2 <= end1:
                    time_intersection = abs(start2 - min(end1, end2))
                elif start2 <= start1 <= end2:
                    time_intersection = abs(start1 - min(end1, end2))

                hours, rest = divmod(time_intersection, 3600)
                minutes, seconds = divmod(rest, 60)
                patient_name = Patient.query.get(record.patient_id).username
                if hours < 1:
                    result.loc[c] = [patient_name, person_name,
                                     place_name1, place_name2,
                                     dist, f"{int(minutes)} : {seconds:.2f}",
                                     human_time_start, human_time_end]
                    c += 1
    return result


def read_data_file(file_path, person_name, patient=False):
    if person_name.lower() in [pt.username for pt in Patient.query.all()]:
        flash(
            f"Patient by the name: {person_name} already exists in database.",
            "warning")
        return

    current_patient = Patient(username=person_name)

    if patient:
        db.session.add(current_patient)
        db.session.commit()

    with open(file_path) as f:
        data = json.load(f)

    if not patient:
        all_records = Record.query.all()
        results = find_timeline_intersection(person_name, data, all_records)
        return results

    for activity in data['timelineObjects']:
        if 'placeVisit' in activity.keys():
            start_long = round(float(activity['placeVisit']['location']['longitudeE7'])/1e7, 7)
            start_lat = round(float(activity['placeVisit']['location']['latitudeE7'])/1e7, 7)
            if 'name' in activity['placeVisit']['location'].keys():
                place_name = activity['placeVisit']['location']['name']
            else:
                continue
            if 'address' in activity['placeVisit']['location'].keys():
                address = activity['placeVisit']['location']['address']
            else:
                continue
            address = ' '.join(address.split())
            duration_start = int(
                activity['placeVisit']['duration']['startTimestampMs']) / 1e3
            duration_end = int(
                activity['placeVisit']['duration']['endTimestampMs']) / 1e3
            du_start = time.strftime(
                '%Y-%m-%d %H:%M:%S',
                time.localtime(
                    duration_start))
            du_end = time.strftime(
                '%Y-%m-%d %H:%M:%S',
                time.localtime(
                    duration_end))
            record = Record(longitude=start_long, latitude=start_lat,
                            place_name=place_name, time_start=du_start,
                            time_end=du_end, raw_time_start=duration_start,
                            raw_time_end=duration_end, patient_id=current_patient.id)
            if patient:
                db.session.add(record)

    if patient:
        db.session.commit()
        flash(f"Patient Record uploaded to database for {person_name}", "success")


@app.route("/")
def home_page():
    return render_template('home.html')


@app.route("/timeline/", methods=['GET', 'POST'])
def contact_trace():
    form = PatientEntryForm()
    suspect_form = SuspectEntryForm()

    if form.validate_on_submit() and form.submit.data:
        file = form.file.data

        if file.filename == '':
            print("Your file must have a filename")
            return redirect(request.url)

        if not allowed_data_files(file.filename):
            print("That file extension is not allowed")
            return redirect(request.url)
        else:
            filename = secure_filename(file.filename)
            the_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(the_path)
            person_name = form.username.data
            read_data_file(the_path, person_name.lower(), patient=True)

        return redirect(request.url)

    if suspect_form.validate_on_submit() and suspect_form.submit2.data:
        print("IM HEREEEEEE")
        file = suspect_form.file.data

        if file.filename == '':
            print("Your file must have a filename")
            return redirect(request.url)

        if not allowed_data_files(file.filename):
            print("That file extension is not allowed")
            return redirect(request.url)
        else:
            filename = secure_filename(file.filename)
            the_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(the_path)
            person_name = suspect_form.username.data
            df_intersect = read_data_file(the_path, person_name.lower())
            if not df_intersect.empty:
                return render_template(
                    'intersect.html',
                    column_names=df_intersect.columns.values,
                    row_data=list(
                        df_intersect.values.tolist()),
                    zip=zip)
            else:
                flash(f"No possible matches found. You're safe!", "success")
                return render_template('intersect.html')

        return redirect(request.url)

    return render_template('timeline.html', form1=form, form2=suspect_form)


@app.route("/lungs/")
def template_test():
    return render_template(
        'lungs.html',
        label='',
        imagesource='../uploads/icon.jpeg')


@app.route('/lungs/', methods=['GET', 'POST'])
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

            label = result[0]
            res = str(result[1])
            label = label + "{ Score : " + res + "}"
            print(result)
            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(
                file_path,
                os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    filename))
            print("--- %s seconds ---" % str(time.time() - start_time))
            return render_template(
                'lungs.html',
                label=label,
                imagesource='../uploads/' +
                filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads': app.config['UPLOAD_FOLDER']
})
