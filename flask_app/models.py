from flask_app import db


# Patient Model
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    records = db.relationship('Record', backref='patient', lazy=True)

    def __repr__(self):
        return f"User('{self.username}')"


# Patient's record
class Record(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    longitude = db.Column(db.Float, nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    place_name = db.Column(db.String(100), nullable=False)
    time_start = db.Column(db.String(60), nullable=False)
    time_end = db.Column(db.String(60), nullable=False)
    raw_time_start = db.Column(db.Float, nullable=False)
    raw_time_end = db.Column(db.Float, nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'),
                           nullable=False)

    def __repr__(self):
        return f"Record('{self.place_name}')"
