import os
from flask import Flask
from flask import render_template, request
from t5_inf import RaceInfModule

# Flask App
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["APPLICATION_ROOT"] = os.environ.get("APP_ROOT","/service")

# Model
q_model = RaceInfModule.load_from_checkpoint(app.config["APPLICATION_ROOT"]+"/ckpts/t5_que.ckpt")
q_model.eval()
d_model = RaceInfModule.load_from_checkpoint(app.config["APPLICATION_ROOT"]+"/ckpts/t5_dis.ckpt")
d_model.eval()


@app.route("/")
def index():
    return render_template("index.html",app_root=app.config["APPLICATION_ROOT"])


@app.route("/predict",methods=["POST"])
def predict():
    global q_model, d_model
    
    article=request.json["article"]
    answer = request.json["answer"]
    print("Start generating..")
    question = q_model.generate_sentence(article, answer)
    print("Question generated!")
    distractor = d_model.generate_sentence(article, answer, question)
    print("Distractor generated!")

    generation = {'question': question, 'distractor': distractor}
    generation_html=render_template("result.html",generation=generation)
    return {"generation_html":generation_html}


@app.route("/predict_json",methods=["POST"])
def predict_json():
    global q_model, d_model

    article = request.json["article"]
    answer = request.json["answer"]
    question = q_model.generate_sentence(article, answer)
    distractor = d_model.generate_sentence(article, answer, question)

    generation = {'question': question, 'distractor': distractor}
    return {"generation":generation}
