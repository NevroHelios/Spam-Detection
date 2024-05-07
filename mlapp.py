import torch
import torch.nn as nn
import pickle
from flask import Flask, request, render_template, jsonify

from Preprocess.preprocess import pre_process
from model.model import DetectSpamV0

# path to the saved model
PATH = "model/model_detect_spam_V0.pt"

# get the availaable device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# initialize the model and load it from the path
model = DetectSpamV0().to(device)
model.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
print("Model loaded")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['get'])
def predict():
    mail = request.args.get('mail')
    prediction = ''
    if mail:
        text = pre_process(mail)

        infer = {0:'Spam', 1:'ham'}

        model.eval()
        with torch.inference_mode():
            logits = model(text.to(device)).squeeze()
            label = torch.round(torch.sigmoid(logits))
            
        prediction = infer[label.item()]

    return render_template('prediction.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)