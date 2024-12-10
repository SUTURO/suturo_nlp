#!/home/siwall/venvs/whisper_venv/bin/python3.8

from nlp_gpsr import *
from nlp_mcrs import *
from scipy.io import wavfile
import speech_recognition as sr
import subprocess


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
        This function is used to run the nlp_gpsr.py script with an audio file as input instead of using a microphone.
        """
        return 0


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
        This function is used to run the nlp_mcrs.py script with an audio file as input instead of using a microphone.
        """
        subprocess.call("python nlp_mcrs.py -a", shell=True)

test_mcrs.run_mcrs()