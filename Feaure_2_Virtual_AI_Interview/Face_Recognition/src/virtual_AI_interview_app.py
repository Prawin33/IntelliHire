# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13, 2024

@author: Batool Talha
"""


import numpy as np
import pickle
import pandas as pd
from flask import Flask, request
from flask import Flask, request, jsonify, render_template
import threading
from virtual_AI_interview_main import main_func

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')


def start_main():
    thread = threading.Thread(target=main_func)
    thread.daemon = True
    thread.start()


@app.route('/start_interview')
def start_interview():
    start_main()
    return "Candidate verification started!"


if __name__ == '__main__':
    app.run(debug=True)