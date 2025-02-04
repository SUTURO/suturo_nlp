#!/home/siwall/venvs/whisper_venv/bin/python3.8

from argparse import ArgumentParser
import requests
import speech_recognition as sr
import json
import rospy
import audioop
import collections
import numpy
import threading
from queue import Queue
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData
import spacy
import activate_language_processing.beep as beep # type: ignore
from activate_language_processing.nlp import semanticLabelling # type: ignore
import noisereduce as nr
from nlp_challenges import *
import os

# Flag to enable/disable ROS functionality
USE_ROS = False  # Set to False for testing without ROS

def nluInternal(text, context):
    """
    Process a text input to extract semantic information like intent and entities,
    formats the extracted data, and publishes it as JSON to a ROS topic.

    Args:
        text: The input text to be analyzed.
        context: a dictionary containing several flags and useful variables:
            lock: a threading lock to ensure thread-safe access to shared resources.
            pub: a ROS publisher object to publish processed results to a specified topic.
    """
    with context["lock"]:  # Lock so only one thread may execute this code at a time
        parses = semanticLabelling(text, context)  # Analyze text and return parses (a structured object like a dictionary)
        
        for p in parses:
            # Skip processing if sentence is empty or entities list is empty                
            if not p["sentence"].strip() or not p["entities"]:
                if (p["intent"] != 'affirm' and p["intent"] != "deny") or not p["sentence"].strip():
                    #print(f"[ALP]: Skipping empty or invalid parse. Sentence: '{p['sentence']}', Intent: '{p['intent']}'")
                    continue  

            #print("The sentence is: " + p["sentence"])

            pAdj = {"sentence": p["sentence"], "intent": p["intent"], "entities": []}
            
            # Process entities and define "entities" list in pAdj
            for k, v in p["entities"].items():
                entity_data = v.copy()  # Copy entity’s data dictionary
                entity_data["role"] = v["role"]  # Copy entity’s data dictionary to pAdj under the key corresponding to the role
                entity_data.pop("group")  # Remove metadata that is not needed
                entity_data.pop("idx")  # Remove metadata that is not needed
                pAdj["entities"].append(entity_data)  # Add processed entity to the list

            #switch(pAdj["intent"], json.dumps(pAdj), context)
            
    #print("[ALP]: Done. Waiting for next command.")


def process_audio_file(audio_file_path):
    """
    Process an audio file and return the transcribed text and NLU results.

    Args:
        audio_file_path (str): Path to the audio file.

    Returns:
        str: Transcribed text.
        dict: NLU results (intent and entities).
    """
    r = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = r.record(source)
    
    # Transcribe audio using Whisper
    transcribed_text = r.recognize_whisper(audio, language="english")
    #print(f"Transcribed Text: {transcribed_text}")
    
    # Create context dictionary with required keys
    context = {
        "lock": threading.Lock(),
        "pub": None,  # No ROS publisher for testing
        "nlp": spacy.load("en_core_web_sm"),
        "rasaURI": "http://localhost:5005/model/parse",  # Add this line
        "intent2Roles": {},
        "role2Roles": {},
    }
    
    # Perform NLU processing
    nlu_results = semanticLabelling(transcribed_text, context)
    #print(f"NLU Results: {nlu_results}")
    
    return transcribed_text, nlu_results


def main():
    if USE_ROS:
        # Initialize ROS node
        rospy.init_node('nlp_out', anonymous=True)
        rospy.loginfo("[ALP]: NLP node initialized")

    # Parse command line arguments
    parser = ArgumentParser(prog='activate_language_processing')
    parser.add_argument('-hsr', '--useHSR', action='store_true', help='Flag to record from HSR microphone via the audio capture topic. If you prefer to use the laptop microphone, or directly connect to the microphone instead, do not set this flag.')
    parser.add_argument('-a', '--useAudio', default="./", help="Use an audio file instead of a microphone. Takes the path to an audio file as argument.")
    parser.add_argument('-nlu', '--nluURI', default='http://localhost:5005/model/parse', help="Link towards the RASA semantic parser. Default: http://localhost:5005/model/parse")
    parser.add_argument('-i', '--inputTopic', default='/nlp_test', help='Topic to send texts for the semantic parser, useful for debugging that part of the pipeline. Default: /nlp_test')
    parser.add_argument('-o', '--outputTopic', default='/nlp_out', help="Topic to send semantic parsing results on. Default: /nlp_out")
    parser.add_argument('-stt', '--speechToTextTopic', default='whisper_out', help="Topic to output whisper speech-to-text results on. Default: /whisper_out")
    parser.add_argument('-t', '--terminal', action='store_true', help='Obsolete, this parameter will be ignored: will ALWAYS listen to the input topic.')
    args, unknown = parser.parse_known_args(rospy.myargv()[1:])

    if USE_ROS:
        nlpOut = rospy.Publisher(args.outputTopic, String, queue_size=16)
        stt = rospy.Publisher(args.speechToTextTopic, String, queue_size=1)
    else:
        nlpOut = None
        stt = None

    queue_data = Queue()  # Queue to store the audio data.
    lock = threading.Lock()  # Lock to ensure thread-safe access to shared resources.

    context = {
        "data": numpy.array([], dtype=numpy.int16),
        "useHSR": args.useHSR,
        "useAudio": args.useAudio,
        "audio": args.useAudio,     
        "transcriber": None,
        "listening": False,
        "speaking": False,
        "queue": queue_data,
        "lock": lock,
        "pub": nlpOut,
        "stt": stt,
        "rasaURI": args.nluURI,
        "nlp": spacy.load("en_core_web_sm"),
        "intent2Roles": {},
        "role2Roles": {},
    }

    if USE_ROS:
        if args.useHSR:
            # Subscribe to the audio topic to get the audio data from HSR's microphone
            rospy.Subscriber('/audio/audio', AudioData, lambda msg: record_hsr(msg, context))

        # Subscribe to the nlp_test topic, which allows sending text directly to this node e.g. from the command line.
        rospy.Subscriber("/nlp_test", String, lambda msg: nluInternal(msg.data, context))

        # Execute record() callback function on receiving a message on /startListener
        rospy.Subscriber('/startListener', String, lambda msg: startListener(msg, context))

        rospy.loginfo("[ALP]: NLP node started")
        rospy.spin()
    else:
        # For testing without ROS
        if args.useAudio != "./":
            transcribed_text, nlu_results = process_audio_file(args.useAudio)
            #print("Transcribed Text:", transcribed_text)
            #print("NLU Results:", nlu_results)
            print("")
        else:
            #print("Usage: python nlp_mcrs.py -a <audio_file_path>")
            print("")

if __name__ == "__main__":
    main()