import numpy as np
import pickle
import pandas as pd
from flask import Flask, request
from flask import Flask, request, jsonify, render_template
import threading
from virtual_AI_interview_main import run_ai  # Update import statement
from virtual_AI_interview_voice_main import main_func_initial_chatbot

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

# Callback function for starting the verification process
def start_verification():
    run_ai("verification")  # Call the run_ai function with "verification" action

# Callback function for starting the interview
def start_interview():
    run_ai("interview")  # Call the run_ai function with "interview" action

@app.route('/start_verification')  # Endpoint for starting the verification process
def start_verification_route():
    thread = threading.Thread(target=start_verification)
    thread.daemon = True
    thread.start()
    return "Verification process started!"

@app.route('/start_interview')  # Endpoint for starting the interview
def start_interview_route():
    thread = threading.Thread(target=start_interview)
    thread.daemon = True
    thread.start()
    return "Virtual AI interview started!"

if __name__ == '__main__':
    app.run(debug=True)
