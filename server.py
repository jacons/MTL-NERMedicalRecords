import torch
from flask import Flask, render_template, request

from Configuration import Configuration
from Evaluation.Predictor import Predictor
from Parsing.parser_utils import parse_args
from Training.MTModel import MTBERTClassification

app = Flask(__name__)

args, _ = parse_args()

conf = Configuration(args)
conf.show_parameters(["bert"])

models = "saved_models/model.a.pt"

id2lab_a = {0: 'B-ACTI', 1: 'B-DISO', 2: 'B-DRUG', 3: 'B-SIGN', 4: 'I-ACTI', 5: 'I-DISO', 6: 'I-DRUG',
            7: 'I-SIGN', 8: 'O'}

id2lab_b = {0: 'B-BODY', 1: 'B-TREA', 2: 'I-BODY', 3: 'I-TREA', 4: 'O'}

model = MTBERTClassification.from_pretrained(conf.bert,
                                             id2label_a=id2lab_a,
                                             id2label_b=id2lab_b)

model.load_state_dict(torch.load(models, map_location=torch.device('cpu')))

if conf.cuda:
    model = model.to(conf.gpu)

predictor = Predictor(conf, model, id2lab_a, id2lab_b)

list_of_result = []


@app.route('/', methods=('GET', 'POST'))
def create():
    if request.method == 'POST':

        sentence = request.form['Sentence']

        if "predict" in request.form and sentence != "":
            tag_pred, mask = predictor.predict(sentence)
            result_ = [*zip(sentence.split(), tag_pred, mask)]
            list_of_result.append(result_)

        elif "clear" in request.form:
            list_of_result.clear()

    return render_template('main.html', list_of_result=list_of_result)
