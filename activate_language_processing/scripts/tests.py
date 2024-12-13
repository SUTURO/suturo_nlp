#!/home/siwall/venvs/whisper_venv/bin/python3.8

import subprocess
from nlp_gpsr import *
import nlp_mcrs 
import os
import sys

audios = ["./audio/Order.wav", "./audio/Order2.wav", "./audio/Order3.wav"]

class test_gpsr:
    """
    Class for testing the nlp_gpsr.py script. The idea is to use various audio inputs, process them with gpsr,
    and then match the output with a ground truth. This can be used to test and improve the nlp_gpsr.py script. 

    Methods:
        run_gpsr(audio_file):
            Run the nlp_gpsr.py script with an audio file as input, instead of using a microphone.
    """
    audio_file = "./audio/BringYellowObject.m4a"

    def run_gpsr():
        """
        This function is used to run the nlp_gpsr.py script with an audio file as input multiple times in a row.
        """
        for audio in audios:
            print(f"Processing {audio}...")

            process = subprocess.Popen(['python', 'nlp_gpsr.py', '-a', audio])

            try:
                process.wait(timeout=10)  
            except subprocess.TimeoutExpired:
                print(f"Timeout expired for {audio}, killing process.")
                process.kill() 

            process.terminate()
            process.wait() 

            print(f"Finished processing {audio}.\n")


class test_mcrs:
    """
    Class for testing the nlp_mcrs.py script. The idea is to use various adio inputs, process them with gpsr and
    then match the output with a ground truth. This can be used to test and improve in the nlp_mcrs.py script. 

    Methods:
        run_mcrs(audio_file):
            Run the nlp_mcrs.py script with an audio file as input, instead of using a microphone.
    """
    def run_mcrs():
        """
        This function is used to run the nlp_mcrs.py script with an audio file as input multiple times in a row.
        """
        for audio in audios:
            print(f"Processing {audio}...")

            process = subprocess.Popen(['python', 'nlp_mcrs.py', '-a', audio])

            try:
                process.wait(timeout=10)  
            except subprocess.TimeoutExpired:
                print(f"Timeout expired for {audio}, killing process.")
                process.kill() 

            process.terminate()
            process.wait() 



