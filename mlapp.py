import torch
import torch.nn as nn
import pickle
from flask import Flask, request, render_template, jsonify

from Preprocess.preprocess import pre_process

PATH = "model/model_detect_spam_V0.pt"

class DetectSpamV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=3000, out_features=2000)
        self.layer2 = nn.Linear(in_features=2000, out_features=1000)
        self.layer3 = nn.Linear(in_features=1000, out_features=100)
        self.layer4 = nn.Linear(in_features=100, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer4(self.relu(self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = DetectSpamV0().to(device)
model.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
print("Model loaded")

VEC_PATH = "Preprocess/tfidf_vectorizer.pkl"
with open(VEC_PATH, 'rb') as file:
    tfidf = pickle.load(file)

mail = "yo guys!"


app = Flask(__name__)

@app.route('/', methods=['get'])
def  predict():
    mail = request.args.get('mail')
    prediction = ''
    if mail:
        text = pre_process(mail)
        text = tfidf.transform([text]).toarray().astype('float32')
        text = torch.from_numpy(text)

        infer = {0:'Spam', 1:'ham'}

        with torch.inference_mode():
            logits = model(text.to(device)).squeeze()
            label = torch.round(torch.sigmoid(logits))
            
        prediction = infer[label.item()]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)