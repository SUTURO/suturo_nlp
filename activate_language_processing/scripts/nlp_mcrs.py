#!/home/simon/venvs/rasa_venv/bin/python3


from argparse import ArgumentParser
import speech_recognition as sr
import json
import sys
import threading
import audioop
import collections
import numpy
import threading
from queue import Queue
from std_msgs.msg import String
#from audio_common_msgs.msg import AudioData
import spacy
#import activate_language_processing.beep as beep # type: ignore
from activate_language_processing.nlp import semanticLabelling # type: ignore
import noisereduce as nr
from nlp_challenges import *
import numpy as np
import librosa
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
import whisper
import soundfile as sf
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String, UInt8MultiArray

AudioMsg = UInt8MultiArray # Define AudioMsg as UInt8MultiArray for ROS2 compatibility
model = whisper.load_model("base")  # Load the Whisper model for transcription

def _isTranscribing(context):
    """
    Checks whether a transcription process is currently active and running. 
    Returns True if a transcriber object exists and is actively running (i.e., the transcription process is ongoing), and False otherwise.

    Args:
        context: a dictionary containing several flags and useful variables
            transcriber: The transcriber object, which is expected to be either a thread or process responsible for transcription.
    """
    return (context["transcriber"] is not None) and (context["transcriber"].is_alive())

def nluInternal(text, temp_fp, context):
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
        #text = replace_text(text, audio)
        names, nouns = nounDictionary(text)  # Ensure unpacking matches the corrected return values
        
        if not names:
            prompt = context.get("transcription_hint", f"The user wants to order one or several of these food or drink items: {' , '.join(nouns)}.")
        elif not nouns:
            prompt = context.get("transcription_hint", f"The user says their name is one of these: {' , '.join(names)}.")
        else:
            prompt = context.get("transcription_hint", f"The user says their name is one of these: {' , '.join(names)}. And they like to drink one of these: {' , '.join(nouns)}.")
        
        # use node logger if available
        if "node" in context and isinstance(context["node"], Node):
            context["node"].get_logger().info(f"Using prompt: {prompt}")
        else:
            print(f"Using prompt: {prompt}")

        result = model.transcribe(temp_fp, initial_prompt=prompt)  # Transcribe the audio file using Whisper with an initial prompt
        text = result["text"]

        parses = semanticLabelling(text, context)  # Analyze text and return parses (a structured object like a dictionary)
        
        for p in parses:
            
            # Skip processing if sentence is empty or entities list is empty                
            if not p["sentence"].strip() or not p["entities"]:
                if (p["intent"] != 'affirm' and p["intent"] != "deny" and p["intent"] != "Callout" and p["intent"] != "Hobbies") or not p["sentence"].strip():
                    if "node" in context and isinstance(context["node"], Node):
                        context["node"].get_logger().info(f"[ALP]: Skipping empty or invalid parse. Sentence: '{p['sentence']}', Intent: '{p['intent']}'")
                    else:
                        print(f"[ALP]: Skipping empty or invalid parse. Sentence: '{p['sentence']}', Intent: '{p['intent']}'")
                    continue  
            
            pAdj = {"sentence": p["sentence"], "intent": p["intent"], "entities": []}  # Create a dictionary and initialize an "entities" list
            print(f"Entity items: {p['entities'].items()}")
            for k, v in p["entities"].items():
                entity_data = v.copy()  # Copy entity’s data dictionary
                entity_data.pop("group")  # Remove metadata that is not needed
                entity_data.pop("idx")  # Remove metadata that is not needed
                pAdj["entities"].append(entity_data)  # Append the processed entity to the "entities" list
            # publish as std_msgs.msg.String in ROS2
            context["pub"].publish(String(data=json.dumps(pAdj)))
            if "node" in context and isinstance(context["node"], Node):
                context["node"].get_logger().info("[ALP]: Done. Waiting for next command.")
            else:
                print("[ALP]: Done. Waiting for next command.")


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
            # Normalize incoming message payloads to raw bytes first (works for bytes, lists of ints, memoryviews, etc.)
            try:
                raw = bytes(data.data)
            except Exception:
                # fallback: try converting elements to ints then to bytes
                raw = bytes([int(x) & 0xFF for x in data.data])
            # accumulating raw data in numpy array (16-bit little-endian signed samples expected)
            context["data"] = numpy.concatenate([context["data"], numpy.frombuffer(raw, dtype=numpy.int16)])
            
            """
            Checks if the length of context["data"] has at least 32,000 samples. Why do we do this?
            Assuming a sample rate of 16,000 Hz, 32,000 samples would represent 2 seconds of audio. By waiting until the 
            data array has 2 seconds' worth of audio, the we accumulate enough data to process.     
            """
            if len(context["data"]) >= 32000:
                noise_sample = context["data"][:16000] # We extract the first 16.000 points of data to use as reference for noisereduction.
                reduced_noise_data = nr.reduce_noise(y=context["data"], sr=16000, y_noise=noise_sample) # Uses the noise sample to remove backround noise from the entire data.

                context["queue"].put(reduced_noise_data)  # Adds the reduced_noise_data in the context["queue"].
                context["data"] = numpy.array([], dtype=numpy.int16) # Reset the array to be empty.


def startListener(msg, context):
    """
    In case transcription is not already in process: creates a new thread that executes the transcribeFn function when stared

    msg: the message that triggers the start of the listener.
    context: a dictionary containing several flags and useful variables
        lock: a threading lock to ensure thread-safe access to shared resources.
        transcriber: the transcriber thread object, which will be created and started if transcription is not currently active.
    """
    if "node" in context and isinstance(context["node"], Node):
        context["node"].get_logger().info("[ALP] got start signal")
    else:
        print("[ALP] got start signal")
    with context["lock"]:  # Lock so only one thread may execute this code at a time.
        if not _isTranscribing(context):  # Check if transciption is not already in process.
            context["transcriber"] = threading.Thread(target=transcriberFn, args=(context,))  # Create a new thread that executes the transcriberFn function when started
            context["transcriber"].start()  # Start the transcription.

def audio_data_to_numpy(audio_data, target_sr=16000):
    """
    Convert speech_recognition.AudioData to a NumPy float32 waveform.
    
    Args:
        audio_data (speech_recognition.AudioData): Audio from `r.listen()`.
        target_sr (int): Target sample rate (default: 16000, common in speech recognition).
    
    Returns:
        np.ndarray: Audio waveform in float32 format (normalized to [-1, 1]).
        int: Sample rate.
    """
    # Get raw audio data as bytes
    raw_data = audio_data.get_raw_data()
    
    # Convert to NumPy array (int16)
    audio_array = np.frombuffer(raw_data, dtype=np.int16)
    
    # Convert to float32 and normalize to [-1, 1]
    waveform = librosa.util.buf_to_float(audio_array, dtype=np.float32)
    
    # Resample if needed
    if audio_data.sample_rate != target_sr:
        waveform = librosa.resample(
            waveform,
            orig_sr=audio_data.sample_rate,
            target_sr=target_sr
        )
    
    return waveform, target_sr

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
    r = sr.Recognizer()
    r.pause_threshold = 1.0

    if context["useHSR"]:
        if "node" in context and isinstance(context["node"], Node):
            context["node"].get_logger().info("Waiting for the beep...")
        else:
            print("Waiting for the beep...")
        with context["lock"]:
            context["listening"] = True
        audio = listen2Queue(context["queue"], r)
        with context["lock"]:
            context["listening"] = False
            context["data"] = np.array([], dtype=np.int16)
            context["queue"] = Queue()
    elif context["audio"] == "./":
        with context["lock"]:
            context["listening"] = True
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=1)
            if "node" in context and isinstance(context["node"], Node):
                context["node"].get_logger().info("Speak now...")
            else:
                print("Speak now...")
            audio = r.listen(source)
        with context["lock"]:
            context["listening"] = False
    elif context["useAudio"]:
        audio_path = context["audio"]
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)
        audio = audio_path
    else:
        raise ValueError("Invalid audio source configuration")

    if "node" in context and isinstance(context["node"], Node):
        context["node"].get_logger().info("[Whisper]: Processing...")
    else:
        print("[Whisper]: Processing...")
    if isinstance(audio, sr.AudioData):
        waveform, sr_val = audio_data_to_numpy(audio)
        temp_fp = "/tmp/audio.wav"
        sf.write(temp_fp, waveform, sr_val)
    else:
        temp_fp = str(audio)

    #prompt = context.get("transcription_hint", "The user might be talking about food, service or greetings.")
    result = model.transcribe(temp_fp, language="en")  # Transcribe the audio file using Whisper
    result = result["text"]
    if "node" in context and isinstance(context["node"], Node):
        context["node"].get_logger().info("[Whisper]: Done")
    else:
        print("[Whisper]: Done")
    print(f"\nWhisper result : {result}")
    context["stt"].publish(String(data=result))
    nluInternal(result, temp_fp, context)

def listen2Queue(soundQueue: Queue, rec: sr.Recognizer, startSilence=2, sampleRate=16000, phraseTimeLimit=None, context=None) -> sr.AudioData:
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

    
    while elapsed_time < startSilence:  # Reads audio buffers for a duration of startSilence seconds.
        buffer, soundDuration, energy = getNextBuffer(soundQueue, sampleRate, sampleWidth)
        adjustEnergyLevel(rec, soundDuration, energy) # Adjusts the recognizer's energy threshold to the ambient noise level using adjustEnergyLevel
        elapsed_time += soundDuration

    # Use node logger if provided, otherwise fallback to print
    if context and "node" in context and isinstance(context["node"], Node):
        context["node"].get_logger().info("Say something (using hsr microphone)!")
    else:
        print("Say something (using hsr microphone)!")
    
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
    # Initialize ROS2 and create a node
    rclpy.init(args=sys.argv)
    node = Node('nlp_out')
    node.get_logger().info("[ALP]: NLP node initialized")
 
    # Parse command line arguments
    parser = ArgumentParser(prog='activate_language_processing')
    parser.add_argument('-hsr', '--useHSR', action='store_true', help='Flag to record from HSR microphone via the audio capture topic. If you prefer to use the laptop microphone, or directly connect to the microphone instead, do not set this flag.')
    parser.add_argument('-a', '--useAudio', default="./", help="Use an audio file instead of a microphone.Takes the path to an audio file as argument.")
    parser.add_argument('-nlu', '--nluURI', default='http://localhost:5005/model/parse', help="Link towards the RASA semantic parser. Default: http://localhost:5005/model/parse")
    parser.add_argument('-i', '--inputTopic', default='/nlp_test', help='Topic to send texts for the semantic parser, useful for debugging that part of the pipeline. Default: /nlp_test')
    parser.add_argument('-o', '--outputTopic', default='/nlp_out', help="Topic to send semantic parsing results on. Default: /nlp_out")
    parser.add_argument('-stt', '--speechToTextTopic', default='whisper_out', help="Topic to output whisper speech-to-text results on. Default: /whisper_out")
    parser.add_argument('-t', '--terminal', action='store_true', help='Obsolete, this parameter will be ignored: will ALWAYS listen to the input topic.')
    args, unknown = parser.parse_known_args(sys.argv[1:])
 
    audio = args.useAudio
 
    qos = QoSProfile(depth=10)
    nlpOut = node.create_publisher(String, args.outputTopic, qos)
    rasaURI = args.nluURI
    stt = node.create_publisher(String, args.speechToTextTopic, qos)
    intent2Roles = {}
 
    queue_data = Queue() # Queue to store the audio data.
    lock = threading.Lock() # Lock to ensure that the record_hsr callback does not interfere with the record callback.
 
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
        "node": node,
        "rasaURI": rasaURI,
        "nlp": spacy.load("en_core_web_sm"),
        "intent2Roles": intent2Roles,
        "role2Roles": {},
    }
 
    if args.useHSR:
        # Subscribe to the audio topic to get the audio data from HSR's microphone
        node.create_subscription(AudioMsg, '/audio/audio', lambda msg: record_hsr(msg, context), qos)
 
    # Subscribe to the nlp_test topic, which allows sending text directly to this node e.g. from the command line. 
    node.create_subscription(String, "/nlp_test", lambda msg: nluInternal(msg.data, "./", context), qos)
 
    # Execute record() callback function on receiving a message on /startListener
    node.create_subscription(String, '/startListener', lambda msg: startListener(msg, context), qos)
 
    node.get_logger().info("[ALP]: NLP node started")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
 
if "__main__" == __name__:
    main()
