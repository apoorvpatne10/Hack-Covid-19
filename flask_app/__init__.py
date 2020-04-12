from flask import Flask
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'flask_app/uploads'
app.config['SECRET_KEY'] = '1b706efb9c107f1427893320fca12686'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['ALLOWED_FILE_TYPES'] = ['JSON', 'TXT']
db = SQLAlchemy(app)


from flask_app import routes
