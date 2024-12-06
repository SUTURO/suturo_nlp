#!/home/siwall/venvs/whisper_venv/bin/python3.8

from nlp_gpsr import *
from nlp_mcrs import *
from scipy.io import wavfile
import speech_recognition as sr
import speech_recognition as sr

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
        return 0


def transcriberFn():
    r = sr.Recognizer()
    with sr.AudioFile('./audio/BringYellowObject.wav') as source:
        audio = r.record(source) 
        
    try:
        result = r.recognize_whisper(audio, language="english")  # Process the audio using the Whisper model
        print(f"\nThe whisper result is: {result}")
        return(result)
    except sr.UnknownValueError:
        print("Whisper could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Whisper service; {e}")


transcriberFn()