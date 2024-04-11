# virtual_AI_interview_app.py

import numpy as np
import pickle
import pandas as pd
from flask import Flask, render_template
import threading
from time import sleep  # Import sleep function for delay
from virtual_AI_interview_main import start_interview, start_verification # verification_completed

app = Flask(__name__, template_folder='templates')

def get_verification_status():
    try:
        with open("verification_status.txt", "r") as f:
            status = f.read().strip()
            return status == "True"
    except FileNotFoundError:
        return False  # Default to False if file not found

@app.route('/')
def home():
    verification_completed = get_verification_status()
    print(verification_completed)
    return render_template('index.html', verification_completed=verification_completed, cache_control='no-cache, no-store, must-revalidate')

# Routes for starting the verification and interview processes
@app.route('/start_verification', methods=['GET'])
def start_verification_route():
    thread = threading.Thread(target=start_verification)
    thread.daemon = True
    thread.start()
    return "Verification process started!"

@app.route('/verification_completed', methods=['GET'])
def verification_completed_route():
    if get_verification_status():
        msg = "Candidate Verified Successfully" #  \n Virtual AI interview can be started now!
        return msg
    else:
        return "Verification not completed yet."

@app.route('/start_interview', methods=['GET'])
def start_interview_route():
    if get_verification_status():
        start_interview()
        return "Virtual AI interview started!"
    else:
        return "Virtual AI interview not completed yet."

if __name__ == '__main__':
    app.run(debug=True)
