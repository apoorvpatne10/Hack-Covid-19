from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length
from flask_wtf.file import FileField, FileRequired


class PatientEntryForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=4, max=30)])
    file = FileField('file', validators=[FileRequired()])
    submit = SubmitField('Submit')


class SuspectEntryForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=4, max=30)])
    file = FileField('file', validators=[FileRequired()])
    submit2 = SubmitField('Submit')
