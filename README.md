# halfbloodprince16

Source code for Hack Covid`19.

## Package Installtions:

`pip install -r requirements.txt`


## Steps to Implement
  1. Locate your terminal at app.py file
  2. Make sure that the models dir has the model weights named as "stage-2" in models directory (Eg path: "models/stage-2.pth"). If not then download the model weights from [here](https://www.kaggle.com/halfbloodprince16/covid19/output).
  3. For loading the fastai model we do require to load model with same params and data which we have used while training.
  4. So download the Image dataset with the training.csv file to read the classified file from Image dir. Eg: path "flask_app/covid19-dataset/images".
  5. Till the above steps, we are done loading our model now. Run the app.py file using `python3 app.py`.
  6. Wait for a few seconds while model loads. It's a one time job whenever we restart our server.
  7. Now open the host ip http://0.0.0.0:3000
  8. There it is! A live demonstration of Covid`19 Lung Xray Infection Detection.
