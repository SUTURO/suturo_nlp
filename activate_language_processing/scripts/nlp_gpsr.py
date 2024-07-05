#!/usr/bin/env python3

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
import activate_language_processing.beep as beep
from activate_language_processing.nlp import semanticLabelling

def _isTranscribing(context):
    return (context["transcriber"] is not None) and (context["transcriber"].is_alive())

def nluInternal(text, context):
    with context["lock"]:
        parses = semanticLabelling(text, context)
        print(parses)
        for p in parses:
            pAdj = {"sentence": p["sentence"], "intent": p["intent"]}
            for k, v in p["entities"].items():
                role=v["role"]
                pAdj[role] = v.copy()
                pAdj[role].pop("role")
                pAdj[role].pop("group")
                pAdj[role].pop("idx")
            context["pub"].publish(json.dumps(pAdj))
    rospy.loginfo("[ALP]: Done. Waiting for next command.")
''' TODO: not all these roles are currently recognized. In particular, attribute-like roles are not recognized.
            "object-name": ([(x.get("value")) for x in item.get("entities", []) if x.get("entity") in ["PhysicalArtifact", "drink", "food"]] or [""])[0],
            "object-type": "",  # ?
            "person-name": ([(x.get("value")) for x in item.get("entities", []) if x.get("entity") == "NaturalPerson"] or [""])[0],
            "person-type": "",  # ?
            "object-attribute": "",  # filter attributes with spaCy
            "person-action": "",  # waving?
            "color": "",  # filter attributes with spaCy
            "number": "",  # spaCy can do that
            "from-location": ([(x.get("value")) for x in item.get("entities", []) if x.get("entity") == "PhysicalPlace"] or [""])[0], # does not filter from/to yet
            "to-location": "",
            "from-room": "",
            "to-room": ""
'''
    
def record_hsr(data, context):
    '''
    Callback function for the /audio/audio subscriber to use HSR's microphone for recording. 
    Accumulates a numpy array with the recieved AudioData and puts it into a queue which 
    gets processed by whisper an rasa.

    Args:
        data: AudioData recieved from ros topic /audio/audio
        context: a dictionary containing several flags and useful variables
            queue: Queue to store the audio data.
            data: Accumulated audio data from HSR's microphone.
            lock: Lock to ensure that the record_hsr callback does not interfere with the record callback.
            transcriber: thread to perform sound to text
    '''
    with context["lock"]:
        if context["listening"]:
            # accumulating raw data in numpy array
            context["data"] = numpy.concatenate([context["data"], numpy.frombuffer(bytes(data.data), dtype=numpy.int16)])
            # put accumulated data into queue when it reaches a certain size
            if len(context["data"]) >= 32000:
                context["queue"].put(context["data"])
                context["data"] = numpy.array([], dtype=numpy.int16) # reset the array

def startListener(msg, context):
    rospy.loginfo("[ALP] got start signal")
    with context["lock"]:
        if not _isTranscribing(context):
            context["transcriber"] = Thread(target=transcriberFn, args=(context,))
            context["transcriber"].start()

def transcriberFn(context):
    r = sr.Recognizer() # speech_recognition.Recognizer
    r.pause_threshold = 1.5 # seconds
    
    if context["useHSR"]:
        rospy.loginfo("Wait for the beep, then say something into the HSR microphone!")
        with context["lock"]:
            context["listening"] = True
        audio = listen2Queue(context["queue"], r)
        audio = listen2Queue(context["queue"], r)
        audio = listen2Queue(context["queue"], r)
        with context["lock"]:
            context["listening"] = False
            context["data"] = numpy.array([], dtype=numpy.int16)
            context["queue"] = Queue()
    else:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, 2)
            rospy.loginfo("Say something into the BACKPACK microphone!")
            audio = r.listen(source)

    # Use sr Whisper integration
    rospy.loginfo("[Whisper]: processing...")
    result = r.recognize_whisper(audio, language="english")
    rospy.loginfo("[Whisper]: done")

    print(f"\n The whisper result is: {result}")
    context["stt"].publish(result)
    nluInternal(result, context)

def listen2Queue(soundQueue: Queue, rec: sr.Recognizer, startSilence=2, sampleRate=16000, phraseTimeLimit=None) -> sr.AudioData:
    '''
    Dirty hack to implement some nice functionality of speech_recognition on a data stream
    obtained via a ros topic. 
    (TODO: this would more elegantly be implemented via subclassing speech_recognition.AudioSource)

    Args:
        soundQueue: a queue where elements are binary buffers where each buffer has raw audio data
                    in little endian 16bit/sample format. Blocking read attempts will be done to 
                    this queue, so it is expected that some other source, e.g. a subscriber
                    callback, will feed data into it
        rec: speech_recognition.Recognizer used to call various sound processing functionality
        startSilence: in seconds, the minimum time before speech starts
        sampleRate: in hertz, how many samples in a second
        phraseTimeLimit: None or a maximum duration, in seconds, for a phrase recording

    Returns:
        a speech_recognition.AudioData which contains a recording of speech.
    '''
    def soundLen(buffer, sampleRate):
        return (len(buffer) + 0.0) / sampleRate

    def getNextBuffer(soundQueue, sampleRate, sampleWidth):
        buffer = soundQueue.get()
        soundQueue.task_done()
        soundDuration = soundLen(buffer, sampleRate)
        energy = audioop.rms(buffer, sampleWidth)
        return buffer, soundDuration, energy

    def adjustEnergyLevel(rec, soundDuration, energy):
        # dynamically adjust the energy threshold using asymmetric weighted average
        damping = rec.dynamic_energy_adjustment_damping ** soundDuration
        target_energy = energy * rec.dynamic_energy_ratio
        rec.energy_threshold = rec.energy_threshold * damping + target_energy * (1 - damping)

    sampleWidth = 2
    # Step 1: adjust to noise level
    # Assumes speech is preceded by at least <startSilence> seconds silence. Loops through this interval
    # to adjust an energy threshold that will subsequently be used to detect speech start.
    elapsed_time = 0
    seconds_per_buffer = 0
    while elapsed_time < startSilence:
        buffer, soundDuration, energy = getNextBuffer(soundQueue, sampleRate, sampleWidth)
        adjustEnergyLevel(rec, soundDuration, energy)
        elapsed_time += soundDuration

    rospy.loginfo("Say something (using hsr microphone)! %s %s" % (os.path.realpath(__file__), beep.__file__))
    # Step 2: wait for speech to begin
    beep.SoundRequestPublisher().publish_sound_request()
    # If the energy level exceeds the threshold, consider speech started
    frames = collections.deque()
    frameTime = 0
    while True:
        buffer, soundDuration, energy = getNextBuffer(soundQueue, sampleRate, sampleWidth)
        frames.append((soundDuration, buffer))
        frameTime += soundDuration
        # detect whether speaking has started on audio input
        if energy > rec.energy_threshold: break
        while frameTime > rec.non_speaking_duration:
            d, _ = frames.popleft()
            frameTime -= d
        # dynamically adjust the energy threshold using asymmetric weighted average
        if rec.dynamic_energy_threshold:
            adjustEnergyLevel(rec, soundDuration, energy)

    # At this step, frames contains a list of buffers, and the length of time these buffers recorded is given in
    # frameTime. At this moment, speech should just begun, nonetheless some initial silence is good to keep.
    # Step 3: keep adding to the recorded speech until a long enough pause is detected.
    pauseTime = 0
    while True:
        buffer, soundDuration, energy = getNextBuffer(soundQueue, sampleRate, sampleWidth)
        frames.append((soundDuration, buffer))
        frameTime += soundDuration
        # handle phrase being too long by cutting off the audio
        if phraseTimeLimit and frameTime > phraseTimeLimit:
            break
        # check if speaking has stopped for longer than the pause threshold on the audio input
        if energy > rec.energy_threshold:
            pauseTime = 0
        else:
            pauseTime += soundDuration
        if pauseTime > rec.pause_threshold:  # end of the phrase
            break
    frame_data = b"".join([x[1] for x in frames])
    return sr.AudioData(frame_data, sampleRate, sampleWidth)

def main():
    # Initialize ros node
    rospy.init_node('nlp_out', anonymous=True)
    rospy.loginfo("[ALP]: NLP node initialized")

    # Parse command line arguments
    parser = ArgumentParser(prog='activate_language_processing')
    parser.add_argument('-hsr', '--useHSR', action='store_true', help='Flag to record from HSR microphone via the audio capture topic. If you prefer to use the laptop microphone, or directly connect to the microphone instead, do not set this flag.')
    parser.add_argument('-nlu', '--nluURI', default='http://localhost:5005/model/parse', help="Link towards the RASA semantic parser. Default: http://localhost:5005/model/parse")
    parser.add_argument('-i', '--inputTopic', default='/nlp_test', help='Topic to send texts for the semantic parser, useful for debugging that part of the pipeline. Default: /nlp_test')
    parser.add_argument('-o', '--outputTopic', default='/nlp_out', help="Topic to send semantic parsing results on. Default: /nlp_out")
    parser.add_argument('-stt', '--speechToTextTopic', default='whisper_out', help="Topic to output whisper speech-to-text results on. Default: /whisper_out")
    parser.add_argument('-t', '--terminal', action='store_true', help='Obsolete, this parameter will be ignored: will ALWAYS listen to the input topic.')
    args, unknown = parser.parse_known_args(rospy.myargv()[1:])

    nlpOut = rospy.Publisher(args.outputTopic, String, queue_size=16)
    rasaURI = args.nluURI
    stt = rospy.Publisher(args.speechToTextTopic, String, queue_size=1)
    intent2Roles = {}

    queue_data = Queue() # Queue to store the audio data.
    lock = threading.Lock() # Lock to ensure that the record_hsr callback does not interfere with the record callback.

    context={"data": numpy.array([], dtype=numpy.int16), "useHSR": args.useHSR, "transcriber": None, "listening": False, "speaking": False, "queue": queue_data, "lock": lock, "pub": nlpOut, "stt": stt, "rasaURI": rasaURI, "nlp": spacy.load("en_core_web_sm"), "intent2Roles": intent2Roles, "role2Roles": {}}

    if args.useHSR:
        # Subscribe to the audio topic to get the audio data from HSR's microphone
        rospy.Subscriber('/audio/audio', AudioData, lambda msg: record_hsr(msg, context))

    # Subscribe to the nlp_test topic, which allows sending text directly to this node e.g. from the command line. 
    rospy.Subscriber("/nlp_test", String, lambda msg : nluInternal(msg.data, context))

    # Execute record() callback function on receiving a message on /startListener
    rospy.Subscriber('/startListener', String, lambda msg : startListener(msg, context))

    rospy.loginfo("[ALP]: NLP node started")
    rospy.spin()

if "__main__" == __name__:
    main()

