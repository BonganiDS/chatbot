# Chatbot Deployment with Flask and JavaScript

I'm just attempting to walk you through how to deploy the chatbot that I named Danai

## Step 1:
This repo currently contains the starter files.

Clone repo and create a virtual environment(in brackets I'll have the mac commands)
```
$ git clone https://github.com/BonganiDS/chatbot.git
$ cd chatbot
$ python3 -m venv venv (conda create --name venv python=3) 
$ . venv/bin/activate (source activate venv)
```
## Step 2
 Install dependencies
```
$ (venv) pip install Flask torch torchvision nltk
Install nltk package

$ (venv) python
>>> import nltk
>>> nltk.download('punkt')
```

## Step3
Run

### 1.
```
$ (venv) python train.py
```
This will dump data.pth file. In essence you create the model for the "intelligent" understanding of the question.

### 2.
```
$ (venv) python chat.py
```
The above starts the terminal chatbot that you use to see functionality.

### 3.
```
$ (venv) python app.py
```
This starts up the simple java script UI I made for the chatbot to make it look a bit fancy but it's the same process that's happening when you run chat.py
