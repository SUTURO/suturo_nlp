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
    if not test_mode:
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
    if test_mode:
        result = data.data
    print(f"\n The whisper result is: {result}")

    # Split Sentence into partials for handling multiple intents
    partials = partial_builder(result)

    # send result to RASA
    ans = [requests.post(server, data=bytes(json.dumps({"text": item}), "utf-8")) for item in partials]

    # load answer from RASA 
    response = [json.loads(item.text) for item in ans]

    # change response format
    response = {
    "sentences": [
        {
            "text": item.get("text", ""),
            "intent": item.get("intent", {}).get("name"),
            "entities": {(x.get("entity"), x.get("value")) for x in item.get("entities", [])}
        }
        for item in response
        ]
    }
    # print(response) # debug print
    # Assign variables for better readability
    sentences = response.get("sentences")
    intent = sentences[0].get("intent")

    # Check length of sentence list for multi-intents and filter "order" and "receptionist" to make sure it gets recognized correctly
    # if len(sentences) == 1 or 
    if intent in ["Order", "Receptionist", "affirm", "deny"]:
        switch(sentences[0], sentences[0])
    else:
        multi(response)

def switch(intent, response):
    '''
    Manual Implementation of switch(match)-case because python3.10 first implemented one, this uses 3.8.
    
    Args:
        intent: The intent parsed from the response
        response: The formatted .json from the record function

    Returns:
        The function corresponding to the intent
    '''
    intent = intent.get("intent")
    return {
        "Receptionist": lambda: receptionist(response),
        "affirm": lambda: nlpOut.publish(f"<CONFIRM>, True"),
        "deny": lambda: nlpOut.publish(f"<CONFIRM>, False"),
        "Order": lambda:  order(response)
    }.get(intent, lambda: nlpOut.publish("<NONE>"))()

def receptionist(response):
    '''
    Function for the receptionist task. 

    Args:
        response: Formatted .json from record function.
    '''
    # Setting up all the variables
    data = json.loads(getData(response))

    name = data.get("names")
    name = name[0] if name else None

    drink = data.get("drinks")
    drink = drink[0] if drink else None

    nlpOut.publish(f"<GUEST>, {name}, {drink}")

def order(response):
    '''
    Function for the Restaurant task.

    Args:
        response: Formatted .json from the record function.
    '''
    data = json.loads(getData(response))

    drinks = data.get("drinks")
    foods = data.get("foods")
    nlpOut.publish(f"<ORDER>, {drinks}, {foods}")

def multi(responses):
    '''
    Handle multiple intents at once.
    
    Args:
        responses: List of rasa responses
    '''
    # Build lists of people, places and artifacts using the rasa responses.
    person_list, place_list, artifact_list = [], [], []
    for sentence in responses["sentences"]:
        tperson_list, tplace_list, tartifact_list = [], [], []
        entities = sentence["entities"]
        for name, value in entities:
            if name == "NaturalPerson":
                tperson_list.append(value)
            else:
                tperson_list.append("")
            if name == "PhysicalPlace":
                tplace_list.append(value)
            else:
                tplace_list.append("")
            if name == "PhysicalArtifact":
                tartifact_list.append(value)
            else:
                tartifact_list.append("")
        person_list.append(tperson_list)
        place_list.append(tplace_list)
        artifact_list.append(tartifact_list)

    # Manually remove "then" from place_list because rasa recognizes it as a place
    i = 0
    for l in place_list:
        place_list[i] = filter(lambda word: word != "then", l)
        i+=1

    # Remove duplicates and empty strings using the shorten function
    person_list, place_list, artifact_list = [shorten(i) for i in person_list], [shorten(i) for i in place_list], [shorten(i) for i in artifact_list]

    # Iterate over the whole sentence and exchange pronouns (and adverbs) with the same role from an earlier partial sentence
    counter = 0
    output = []
    for sentence in responses["sentences"]:
        text = sentence["text"]
        words = text.split()
        for word in words:
            if counter > 0: # only replace words from the second sentence onwards
                if  sem[word] == "PRON" or sem[word] == "ADV":
                    # Repeating this for every list
                    if word in place_list[counter]:
                        if place_list[counter-1] == []:
                            word = place_list[0][0] if place_list[0] else word # don't do anything if list is empty
                        else:
                            word = place_list[counter-1][0]
                            place_list[counter].insert(0, word)

                    elif word in person_list[counter]:
                        if person_list[counter-1] == []:
                            word = person_list[0][0] if person_list[0] else word
                        else:
                            word = person_list[counter-1][0]
                            person_list[counter].insert(0, word)

                    elif word in artifact_list[counter]:
                        if artifact_list[counter-1] == []:
                            word = artifact_list[0][0] if artifact_list[0] else word
                        else:
                            word = artifact_list[counter-1][0]
                            artifact_list[counter].insert(0, word)

            output.append(word)
        counter += 1

    # Split the sentence into a list of partials at specific words
    output = split_into_queue(splits, output)

    # calling rasa twice for each partial sentence seems bad practice
    output = [requests.post(server, data=bytes(json.dumps({"text": item}), "utf-8")) for item in output]
    output = [json.loads(item.text) for item in output]

    # Change the format for better usability
    for idx, item in enumerate(output):
        result = {
            "sentence": item.get("text"),
            "intent": item.get("intent", {}).get("name"),
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
        }

        print(str(result))
        nlpOut.publish(str(result))

def partial_builder(sentence):
    '''
    Builds multiple partial sentences from a single sentence.
    
    Args:
        sentence: A string
    
    Returns:
        A List of partial sentences.
    '''
    # list of verbs to ignore when generating partial sentences, all lowercase
    ignore_list = ["order"]
    # list of verbs to always let through, all lowercase
    special_list = ["bring"]

    # Setting up variables for building partials and initiating spaCy 
    doc = nlp(sentence)
    global sem, splits
    temp, sents, sem, splits = "", [], {}, []
    first = True

    # Iterate over the words in the sentence and build up partial sentences by using temporary lists until verbs appear.
    # Ignore gerunds and words from the ignore_list.
    for token in doc:
        sem.update({token.text:token.pos_})
        if token.pos_ == "VERB" and token.text[-3:] != "ing" and str.lower(token.text) not in ignore_list or str.lower(token.text) in special_list:
            if first == True:
                temp = temp + token.text + " "
                first = False
            else:
                sents.append(temp)
                temp = token.text + " "
                splits.append(token.text)
        else:
            temp = temp + token.text + " "
    sents.append(temp)
    # sents = list(filter(None, sents))
    return sents

def shorten(array):
    '''
    Function that removes duplicates and empty Strings

    Args:
        array: Any list
    Returns:
        A List without duplicates and no empty Strings
    '''
    # Sets don't allow duplicates
    s = set(array)
    if "" in s:
        s.remove("")    # remove empty string
    return list(s)

def split_into_queue(verbs, sent):
    '''
    Function that splits a sentence into parts at specific words.

    Args:
        verbs: A list of Strings
        sent: A String
    Returns:
        A list of partial sentences, split at the strings from verbs
    '''
    temp = ""
    out = []
    # Iterate over all the words in the sentence and split into new sentence on verbs from list
    for word in sent:
        if word in verbs:
            out.append(temp)
            temp = word
        else:
            temp += f" {word}"
    out.append(temp)
    out = list(filter(None, out))   # filter empty strings and make into a list to avoid python iterables
    return out

def getData(data):
    '''
    Function for getting names, drinks and foods from entities in a .json

    Args:
        data: The json data
    Returns:
        The list of entities
    '''
    # Setup variables
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
    # Parse command line arguments
    parser = ArgumentParser(prog='activate_language_processing')
    parser.add_argument('-hsr', '--useHSR', action='store_true', help='Flag to record from HSR microphone via the audio capture topic. If you prefer to use the laptop microphone, or directly connect to the microphone instead, do not set this flag.')
    parser.add_argument('-t', '--terminal', action='store_true', help='Flag to use the "/nlp_test" topic instead of microphone input.')
    args, unknown = parser.parse_known_args(rospy.myargv()[1:])

    queue_data = Queue() # Queue to store the audio data.
    lock = threading.Lock() # Lock to ensure that the record_hsr callback does not interfere with the record callback.
    flags = {"record": False}
    acc_data = {"data": numpy.array([], dtype=numpy.int16)} # Accumulated audio data from HSR's microphone.

    # Manage arguments
    if args.useHSR:
        # Subscribe to the audio topic to get the audio data from HSR's microphone
        rospy.Subscriber('/audio/audio', AudioData, lambda msg: record_hsr(msg, queue_data, acc_data, lock, flags))

    global test_mode
    test_mode = args.terminal
    if args.terminal:
        # Subscribe to the nlp_test topic to 
        rospy.Subscriber("/nlp_test", String, lambda msg : record(msg, args.useHSR, queue_data, lock, flags))
    else:
        # Execute record() callback function on receiving a message on /startListener
        rospy.Subscriber('/startListener', String, lambda msg : record(msg, args.useHSR, queue_data, lock, flags))  # TODO test what happens when 2 signals overlap

    rospy.spin()

if "__main__" == __name__:

    # Initialize ros node
    rospy.init_node('nlp_out', anonymous=True)
    rate = rospy.Rate(1)

    # Initiate nlp_out publisher
    nlpOut = rospy.Publisher("nlp_out", String, queue_size=16)

    # rasa Action server
    server = "http://localhost:5005/model/parse"

    # initiate spaCy
    global nlp
    nlp = spacy.load("en_core_web_sm")

    main()
