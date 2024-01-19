import requests
import speech_recognition as sr
import json
import rospy
from std_msgs.msg import String, Bool
from std_srvs.srv import SaveInfo # TODO should be changed to the actual location

# unique id for every new person
person_id = 1.0

def main():    
    # Execute record() function on receiving a message on /startListener
    rospy.Subscriber('/startListener', String, record)
    rospy.spin()

'''
This function gets the json format from rasa and outputs the drink value
*data* The json data
'''
def getDrink(data):
    entities = data.get("entities")
    for ent, val  in entities:
        if ent == "drink":
            return val
    return None

'''
This function gets the json format from rasa and outputs the NaturalPerson value
'data' The json data
'''
def getName(data):
    entities = data.get("entities")
    for ent, val in entities:
            if ent == "NaturalPerson":
                return val
    return None

'''
This function records from the microphone, sends it to whisper and to the Rasa server
'''
def record(data):  

    # needs to be redefined here otherwise error occures
    global person_id
    
    # Record with speech recognizer for as long as someone is talking
    r = sr.Recognizer()
    r.pause_threshold = 1.5
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, 2)
        print("Say something!")
        audio = r.listen(source)

    # Write recorder audio to file
    with open("temp_file", "wb") as file:
        file.write(audio.get_raw_data())

    # Use sr Whisper integration  
    result = r.recognize_whisper(audio, language="english")

    # send result to RASA
    ans = requests.post(server, data=bytes(json.dumps({"text": result}), "utf-8"))

    # load answer from RASA 
    response = json.loads(ans.text)   

    # change response format
    response = {"text": response.get("text", ), "intent": response.get("intent", {}).get("name"), 
                "entities": set([(x.get("entity"), x.get("value")) for x in response.get("entities", [])])}

    intent = response.get("intent")
    name = str(getName(response))
    drink = str(getDrink(response))

    # service call to knowledge
    if intent == "Receptionist" and name != "None" and drink != "None":
        
        # wait for the knowledge service to be up and running
        rospy.wait_for_service('save_server') 
        # call the knowledge service       
        callService = rospy.ServiceProxy('save_server', SaveInfo)

        try:        
            # call knowledge service with required data
            res = callService(f"{name}, {drink}, {str(person_id)}")
        
            # to give every new recognized person a unique id
            person_id += 1
            
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
            nlpFeedback.publish(False)
            
    elif intent == "Receptionist" and (name == "None" or drink == "None"):
        print("Either Name or Drink was not understood.")
        nlpFeedback.publish(False)
    else:
        print("Did not understand a Receptionist task.")
        nlpFeedback.publish(False)

    

if "__main__" == __name__:

    # Alternative code to create nlp_out topic
    rospy.init_node('nlp_out', anonymous=True)
    nlpFeedback = rospy.Publisher("nlp_feedback", Bool, queue_size=16)
    rate = rospy.Rate(1)

    # rasa Action server
    server = "http://localhost:5005/model/parse" 
    
    main()
