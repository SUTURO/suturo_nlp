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


def _isTranscribing(context):
    """
    Checks whether a transcription process is currently active and running. 
    Returns True if a transcriber object exists and is actively running (i.e., the transcription process is ongoing), and False otherwise.

    Args:
        context: a dictionary containing several flags and useful variables
            transcriber: The transcriber object, which is expected to be either a thread or process responsible for transcription.
    """
    return (context["transcriber"] is not None) and (context["transcriber"].is_alive())

def nluInternal(text, context):
    """
    Process a text input to extract semantic information like intent and entities, 
    formats the extracted data, and publishes it as JSON to a ROS topic.

    Args:
        text: The input text to be analyzed.
        context: a dictionary containing several flags and useful variables
            lock: a threading lock to ensure thread-safe access to shared resources.
            pub: a ROS publisher object to publish processed results to a specified topic.
    """
    with context["lock"]: # Lock so only one thread may execute this code at a time
        parses = semanticLabelling(text, context) # Analyze text and return parses (a structured object like a dictionary)
        print(parses)
        for p in parses:
            pAdj = {"sentence": p["sentence"], "intent": p["intent"]} # create a dictionary and store the sentence and intent extracted from a parse p 
            for k, v in p["entities"].items(): 
                role=v["role"] # Extract the value of a "role" in an entity
                pAdj[role] = v.copy() # Copy entity’s data dictionary to pAdj under the key corresponding to the role 
                pAdj[role].pop("role") # Remove the "role" since its already used as key
                pAdj[role].pop("group") # Remove metadata that is not needed
                pAdj[role].pop("idx") # Remove metadate that is not needed
            context["pub"].publish(json.dumps(pAdj)) # Convert pAdj to JSON string and publish to a rostopic
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
            queue: queue to store the audio data.
            data: accumulated audio data from HSR's microphone.
            lock: lock to ensure that the record_hsr callback does not interfere with the record callback.
            transcriber: thread to perform sound to text
    '''
    with context["lock"]:
        if context["listening"]:
            # accumulating raw data in numpy array
            context["data"] = numpy.concatenate([context["data"], numpy.frombuffer(bytes(data.data), dtype=numpy.int16)])
            
            """
            Checks if the length of context["data"] has at least 32,000 samples. Why do we do this?
            Assuming a sample rate of 16,000 Hz, 32,000 samples would represent 2 seconds of audio. By waiting until the 
            data array has 2 seconds' worth of audio, the we accumulate enough data to process.     
            """
            if len(context["data"]) >= 32000:
                noise_sample = context["data"][:16000] # We extract the first 16.000 points of data to use as reference for noisereduction.
                reduced_noise_data = nr.reduce_noise(y=context["data"], sr=16000, y_noise=noise_sample) # Uses the noise sample to remove backround noise from the entire data.

                context["queue"].put(reduced_noise_data) # Adds the reduced_noise_data in the context["queue"].
                context["data"] = numpy.array([], dtype=numpy.int16) # Reset the array to be empty.


def startListener(msg, context):
    """
    In case transcription is not already in process: creates a new thread that executes the transcribeFn function when stared

    msg: the message that triggers the start of the listener.
    context: a dictionary containing several flags and useful variables
        lock: a threading lock to ensure thread-safe access to shared resources.
        transcriber: the transcriber thread object, which will be created and started if transcription is not currently active.
    """
    rospy.loginfo("[ALP] got start signal")
    with context["lock"]: # Lock so only one thread may execute this code at a time.
        if not _isTranscribing(context): # Check if transciption is not already in process.
            context["transcriber"] = threading.Thread(target=transcriberFn, args=(context,)) # Create a new thread that executes the transcriberFn function when started
            context["transcriber"].start() # Start the transcription.


def transcriberFn(context):
    """
    Responsible for handling the transcription process.
    Captures audio input, transcribes it using the whisper speech recognition model, and then publishes the result.

    Args:
    context: a dictionary containing shared resources and state information required for transcription. It should include:
        lock: a threading lock to ensure thread-safe access to shared resources.
        useHSR: a flag indicating whether to use the HSR microphone (`True`) or the Backpack microphone (`False`).
        queue: a queue used for handling audio data when using the HSR microphone.
        data: an array for storing audio data when using the HSR microphone.
        listening: a flag indicating whether transcription is currently in progress.
        stt: a ROS publisher object used to publish the transcription result to other ROS nodes.
    """
    r = sr.Recognizer() # speech_recognition.Recognizer
    r.pause_threshold = 1.0 # seconds
    
    if context["useHSR"]: # If true, function assumes an HSR-specific microphone setup
        rospy.loginfo("Wait for the beep, then say something into the HSR microphone!")
        with context["lock"]: 
            context["listening"] = True # Transcirption is in progress
        audio = listen2Queue(context["queue"], r) # Capture audio
        with context["lock"]: 
            context["listening"] = False # Transcription is no longer in progress
            context["data"] = numpy.array([], dtype=numpy.int16) # Reset the array to be empty
            context["queue"] = Queue() # Reset the queue to be empty 
    else:
        with context["lock"]:
            context["listening"] = True # Transcirption is in progress
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, 1) # Adjust for noisy environment
            rospy.loginfo("Say something into the BACKPACK microphone!")
            #beep.SoundRequestPublisher().publish_sound_request() # Publish beep sound
            rospy.loginfo("[ALP] listening....")
            audio = r.listen(source) # Recognizer listens to audio
            rospy.loginfo("[ALP] Done listening.")
        with context["lock"]:
            context["listening"] = False # Transcription is no longer in progress

    # Use sr Whisper integration
    rospy.loginfo("[Whisper]: processing...")
    result = r.recognize_whisper(audio, language="english") # Process the audio using the english whisper model to convert speech to text
    rospy.loginfo("[Whisper]: done")

    print(f"\n The whisper result is: {result}")
    context["stt"].publish(result) # Transcription result is published to a rostopic
    nluInternal(result, context) # Call nluInternal to process transcription result


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
        return (len(buffer) + 0.0) / sampleRate # Computes the duration of the audio buffer in seconds based on its length and sample rate.

    def getNextBuffer(soundQueue, sampleRate, sampleWidth):
        buffer = soundQueue.get() # Retrieves the next buffer from the queue
        soundQueue.task_done()
        soundDuration = soundLen(buffer, sampleRate) # Calculate buffer duration
        energy = audioop.rms(buffer, sampleWidth) # Calculate buffer energy level for detecting speech
        return buffer, soundDuration, energy

    def adjustEnergyLevel(rec, soundDuration, energy):
        # dynamically adjust the energy threshold using asymmetric weighted average.
        damping = rec.dynamic_energy_adjustment_damping ** soundDuration
        target_energy = energy * rec.dynamic_energy_ratio
        rec.energy_threshold = rec.energy_threshold * damping + target_energy * (1 - damping)
    
    sampleWidth = 2 # Audio samples are 16-bit (2 bytes/sample)
    
    # Step 1: adjust to noise level
    # Assumes speech is preceded by at least <startSilence> seconds silence. Loops through this interval
    # to adjust an energy threshold that will subsequently be used to detect speech start.
    elapsed_time = 0 #  Tracks total time for adjusting noise levels.
    seconds_per_buffer = 0

    
    while elapsed_time < startSilence: # Reads audio buffers for a duration of startSilence seconds.
        buffer, soundDuration, energy = getNextBuffer(soundQueue, sampleRate, sampleWidth)
        adjustEnergyLevel(rec, soundDuration, energy) # Adjusts the recognizer's energy threshold to the ambient noise level using adjustEnergyLevel
        elapsed_time += soundDuration

    rospy.loginfo("Say something (using hsr microphone)!")
    
    # Step 2: wait for speech to begin
    #beep.SoundRequestPublisher().publish_sound_request()
    # If the energy level exceeds the threshold, consider speech started
    frames = collections.deque()
    frameTime = 0
    while True:
        buffer, soundDuration, energy = getNextBuffer(soundQueue, sampleRate, sampleWidth)
        frames.append((soundDuration, buffer))
        frameTime += soundDuration
        if energy > rec.energy_threshold: break # Wait until the energy of an audio buffer exceeds the threshold, indicating speech has started.
        while frameTime > rec.non_speaking_duration: 
            d, _ = frames.popleft() 
            frameTime -= d # Remove old frames to keep memory small
        if rec.dynamic_energy_threshold:
            adjustEnergyLevel(rec, soundDuration, energy) # dynamically adjust the energy threshold using asymmetric weighted average

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

    frame_data = b"".join([x[1] for x in frames]) # Concatenate all the audio frames in frames into a single byte string called frame_data.
    
    frame_data_array = numpy.frombuffer(frame_data, dtype=numpy.int16) # Convert the now byte string "frame_data" into a NumPy array called frame_data_array.

    noise_reduced_data = nr.reduce_noise(y=frame_data_array, sr=sampleRate) # Use the noisereduce to apply noise reduction on the audio data stored in frame_data_array.
    
    frame_data_clean = noise_reduced_data.tobytes()  # Convert the cleaned audio data in noise_reduced_data back into a byte string format, frame_data_clean.

    return sr.AudioData(frame_data_clean, sampleRate, sampleWidth) # Wrap frame_data_clean in an AudioData object from the speech_recognition library.



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

