# Decepticon - Multiple Choice Question Generation   
 
## Description   
We aim at an end-to-end solution for automatic generation of multiple choice questions for English exams. Creating such exam could be time-consuming, therefore, a model that can suggest question as well as possible choices could potentially speed up the process and improve the exam quality.

## How to run   
First, install dependencies   
```bash
# clone project   
$ git clone https://github.com/vinhng10/decepticon.git

# install project   
$ pip install -e .   
$ pip install -r requirements.txt
 ```   
Next， request the dataset from here
```
https://www.cs.cmu.edu/~glai1/data/race/
```
### Train the model
1. Set the config in the `project/configs` folder
2. `cd project` and `python main.py`

### Use the service(T5 model)

#### Run with script
1. Download the [ckpts](https://drive.google.com/drive/folders/1PQlVkPN3_MKZMvDVj81-u6j4fbozuyp8?usp=sharing) and put them in a right path
2. Run the script
```shell
$ cd project/service
$ python inference.py --article="This is a test" --answer="test"
ANS test
QUE This is a _.
DIS test
```

#### Run with docker
- CPU
```shell
docker run -it -rm -p [port]:8333 hnliu/decepticon:latest 
```
- GPU
```shell
docker run -it -rm -p [port]:8333 hnliu/decepticon:gpu
```

# Decepticon - Multiple Choice Question & Distractors Generation   
Here are the links to the thorough [literature survey](https://github.com/vinhng10/decepticon/blob/master/documents/NLP_Literature_Review_2021.pdf) and [final report](https://github.com/vinhng10/decepticon/blob/master/documents/NLP_Final_Report.pdf) of the project.

## Description   
We aim at an end-to-end solution for automatic generation of multiple choice questions for English exams. Creating such exam could be time-consuming, therefore, a model that can suggest question as well as possible choices could potentially speed up the process and improve the exam quality.

## Dataset
ReAding  Com-prehension Dataset From Examination or RACE (Lai et al. 2017) is used in this project. The dataset consists of middle  and  high  school  exam  multiple-choice  questions.  It  contains  around  28000  unique  articles  and 100000  unique  question-answer  pairs. The  dataset  is  already  split  into  train,  validationand  test  sets.

## Methods
Three models are implemented and experimented in this study: 
1. Seq2Seq model with encoder and decoder implemented with GRU layers.
2. BERT-based model with BERT as encoder and a transformer generation head.
3. Google T5 model fine-tuned with the RACE dataset.

For question generation, we construct the format of the inputs and outputs as follow:
- input: ```[ANS] correct answer tokens [CON] contect tokens```
- target: ```question tokens```

For distractor generation, the format is as follow:
- input: ```[ANS] correct answer tokens [CON] contect tokens [QUE] question tokens```
- target: ```question tokens```

## Evaluation
In this study, we use BLEU, METEOR, and ROUGE scores to automatically measure the performance of the models. Additionally, we conduct qualitative analysis to see if the generated questions and distractors correlate well with human judgement. Please refer to the final report for the evaluation.
