#!/usr/bin/env python3

from argparse import ArgumentParser
import requests
import speech_recognition as sr
import json
import rospy
import audioop
import collections
import queue
import numpy
import threading
from queue import Queue
from std_msgs.msg import String, Bool
from audio_common_msgs.msg import AudioData
from knowledge_msgs.srv import SaveInfo

# unique id for every new person
person_id = 1.0

def record_hsr(data, queue_data, acc_data, lock, flags):
    '''
    Callback function for the /audio/audio subscriber to use HSR's microphone for recording. 
    Accumulates a numpy array with the recieved AudioData and puts it into a queue which 
    gets processed by whisper an rasa.

    Args:
        data: AudioData recieved from ros topic /audio/audio
        queue_data: Queue to store the audio data.
        acc_data: Accumulated audio data from HSR's microphone.
        lock: Lock to ensure that the record_hsr callback does not interfere with the record callback.
        flags: Dictionary to store the record flag.
    '''
    doSomething = False
    with lock:
        doSomething = flags["record"]
    if not doSomething:
        acc_data["data"] = numpy.array([], dtype=numpy.int16)
        return
    # accumulating raw data in numpy array
    acc_data["data"] = numpy.concatenate([acc_data["data"], numpy.frombuffer(bytes(data.data), dtype=numpy.int16)])
    # put accumulated data into queue when it reaches a certain size
    if len(acc_data["data"]) >= 32000:
        queue_data.put(acc_data["data"])    
        acc_data["data"] = numpy.array([], dtype=numpy.int16) # reset the array

def record(data, recordFromTopic, queue_data, lock, flags):
    '''
    This callback function records from the microphone, sends it to whisper and to the Rasa server
    and then processes the response.

    Args:
        data: String message recieved from the /startListener topic
        recordFromTopic: Flag to select the microphone to record from.
        queue_data: Queue to store the audio data.
        lock: Lock to ensure that the record_hsr callback does not interfere with the record callback.
        flags: Dictionary to store the record flag.
    '''
    r = sr.Recognizer() # speech_recognition.Recognizer
    r.pause_threshold = 1.5 # seconds

    if recordFromTopic:
        with lock:
            flags["record"] = True
        # print("Say something (using hsr microphone)!")
        audio = listen2Queue(queue_data, r)
        with lock:
            flags["record"] = False
    else:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, 2)
            print("Say something (using backpack microphone)!")
            audio = r.listen(source)

    # Use sr Whisper integration
    result = r.recognize_whisper(audio, language="english")
    print(f"\n The whisper result is: {result}")

    # send result to RASA
    ans = requests.post(server, data=bytes(json.dumps({"text": result}), "utf-8"))

    # load answer from RASA 
    response = json.loads(ans.text)   

    # change response format
    response = {"text": response.get("text", ), "intent": response.get("intent", {}).get("name"), 
                "entities": set([(x.get("entity"), x.get("value")) for x in response.get("entities", [])])}

    # call the switch function to get the right function for the intent
    switch(response.get("intent"), response)

def switch(case, response):
    '''
    Manual Implementation of switch(match)-case because python3.10 first implemented one, this uses 3.8.
    
    Args:
        case: The intent parsed from the response
        response: The formatted .json from the record function

    Returns:
        The function corresponding to the intent
    '''
    return {
        "Receptionist": lambda: receptionist(response),
        "affirm": lambda: confirmFeedback.publish(True),
        "deny": lambda: confirmFeedback.publish(False),
        "Order": lambda:  order(response)
    }.get(case, lambda: nlpFeedback.publish(False))()

def receptionist(response):
    '''
    Function for the receptionist task. 

    Args:
        response: Formatted .json from record function.
    '''
    # Setting up all the variables
    global person_id

    data = json.loads(getData(response))

    name = data.get("names")
    name = name[0] if name else None

    drink = data.get("drinks")
    drink = drink[0] if drink else None

    # Managing the Knowledge Service Interface
    rospy.wait_for_service('save_server')         
    callService = rospy.ServiceProxy('save_server', SaveInfo)

    if name != None and drink != None:
        try:        
            # call knowledge service with required data
            res = callService(f"{name}, {drink}, {str(person_id)}")
            nlpOut.publish(f"{name}, {drink}, {str(person_id)}")            
            person_id += 1 # to give every new recognized person a unique id
        # if the service does not process the request, print an error    
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
            nlpFeedback.publish(False)
    else:
        print("Either Name or Drink was not understood.")
        nlpFeedback.publish(False)

def order(response):    #TODO logic
    '''
    Function for the Restaurant task.

    Args:
        response: Formatted .json from the record function.
    '''
    nlpOut.publish(response)


def  getData(data):
    '''
    Function for getting names, drinks and foods from entities in a .json

    Args:
        data: The json data
    Returns:
        The list of entities
    '''
    entities = data.get("entities")
    drinks = []
    foods = []
    names = []
    
    # Filtering the entities list for drink, food and NaturalPerson
    for ent, val in entities:
        if ent == "drink":
            drinks.append(val)
        elif ent == "food":
            foods.append(val)
        elif ent == "NaturalPerson":
            names.append(val)
        else:
            pass
    
    # Build the .json
    list = {"names": names, "drinks": drinks, "foods": foods}
    return json.dumps(list)



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
    
    print("Say something (using hsr microphone)!")

    # Step 2: wait for speech to begin
    # If the energy level exceeds the threshold, consider speech started
    frames = collections.deque()
    frameTime = 0
    while True:
        buffer, soundDuration, energy = getNextBuffer(soundQueue, sampleRate, sampleWidth)
        frames.append((soundDuration, buffer))
        frameTime += soundDuration
        while frameTime > rec.non_speaking_duration:
            d, _ = frames.popleft()
            frameTime -= d
        # detect whether speaking has started on audio input
        if energy > rec.energy_threshold: break
        # dynamically adjust the energy threshold using asymmetric weighted average
        if rec.dynamic_energy_threshold:
            adjustEnergyLevel(rec, soundDuration, energy)

    # At this step, frames contains a list of buffers, and the length of time these buffers recorded is given in
    # frameTime. At this moment, speech should not have begun yet, nonetheless some initial silence is good to keep.
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
    # Parse command line arguments
    parser = ArgumentParser(prog='activate_language_processing')
    parser.add_argument('-hsr', '--useHSR', action='store_true', help='Flag to record from HSR microphone via the audio capture topic. If you prefer to use the laptop microphone, or directly connect to the microphone instead, do not set this flag.')
    args, unknown = parser.parse_known_args(rospy.myargv()[1:])

    queue_data = Queue() # Queue to store the audio data.
    lock = threading.Lock() # Lock to ensure that the record_hsr callback does not interfere with the record callback.
    flags = {"record": False} 
    acc_data = {"data": numpy.array([], dtype=numpy.int16)} # Accumulated audio data from HSR's microphone.

    if args.useHSR:
        # Subscribe to the audio topic to get the audio data from HSR's microphone
        rospy.Subscriber('/audio/audio', AudioData, lambda msg: record_hsr(msg, queue_data, acc_data, lock, flags))
    # Execute record() callback function on receiving a message on /startListener
    rospy.Subscriber('/startListener', String, lambda msg : record(msg, args.useHSR, queue_data, lock, flags))  # TODO test what happens when 2 signals overlap
    
    rospy.spin()

if "__main__" == __name__:

    # Initialize ros node
    rospy.init_node('nlp_out', anonymous=True)
    # Publisher for the feedback
    nlpFeedback = rospy.Publisher("nlp_feedback", Bool, queue_size=16)
    confirmFeedback = rospy.Publisher("nlp_confirmation", Bool, queue_size=16)
    rate = rospy.Rate(1)

    # Alternative code to circumvent Knowledge Service. Don't forget to comment the rospy.wait_for_service and callService lines.
    nlpOut = rospy.Publisher("nlp_out", String, queue_size=16)    

    # rasa Action server
    server = "http://localhost:5005/model/parse" 
    
    main()
