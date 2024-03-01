import requests
import speech_recognition as sr
import json
import rospy
from std_msgs.msg import String, Bool
#from knowledge_msgs.srv import SaveInfo

# unique id for every new person
person_id = 1.0

def main():    
    # Execute record() function on receiving a message on /startListener
    rospy.Subscriber('/startListener', String, record)  # TODO test what happens when 2 signals overlap
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
Manual Implementation of switch(match)-case because python3.10 first implemented one, this uses 3.8.
*case* The intent parsed from the response
*response* The formatted .json from the record function
'''
def switch(case, response):
    print(case)
    return {
        "Receptionist": lambda: receptionist(response),
        "ParkingArms" : lambda: dummy(response) # You can say "What's your name" to test this
    }.get(case, lambda: nlpFeedback.publish(False))() # TODO decide if useful to print "Something went wrong" in dedicated function.

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

    # Use sr Whisper integration
    result = r.recognize_whisper(audio, language="english")

    # send result to RASA
    ans = requests.post(server, data=bytes(json.dumps({"text": result}), "utf-8"))

    # load answer from RASA 
    response = json.loads(ans.text)   

    # change response format
    response = {"text": response.get("text", ), "intent": response.get("intent", {}).get("name"), 
                "entities": set([(x.get("entity"), x.get("value")) for x in response.get("entities", [])])}

    switch(response.get("intent"), response)

'''
Function for the receptionist task. 
*response* Formatted .json from record function.
'''
def receptionist(response):
    global person_id

    name = str(getName(response))
    drink = str(getDrink(response))

    # wait for the knowledge service to be up and running
    #rospy.wait_for_service('save_server') 
    # call the knowledge service       
    #callService = rospy.ServiceProxy('save_server', SaveInfo)
    if name != "None" and drink != "None":
        try:        
            # call knowledge service with required data
            #res = callService(f"{name}, {drink}, {str(person_id)}")
            nlpOut.publish(f"{name}, {drink}, {str(person_id)}")
        
            # to give every new recognized person a unique id
            person_id += 1
            
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
            nlpFeedback.publish(False)
    else:
        print("Either Name or Drink was not understood.")
        nlpFeedback.publish(False)

"""
Dummy function to show how other intents might get implemented.
*response* Formatted .json from the record function.
"""
def dummy(response):
    print(f"Dummy function {response}")
    

if "__main__" == __name__:

    # Initialize ros node
    rospy.init_node('nlp_out', anonymous=True)
    nlpFeedback = rospy.Publisher("nlp_feedback", Bool, queue_size=16)
    rate = rospy.Rate(1)

    # Alternative code to circumvent Knowledge Service. Don't forget to comment the rospy.wait_for_service and callService lines.
    nlpOut = rospy.Publisher("nlp_out", String, queue_size=16)

    # rasa Action server
    server = "http://localhost:5005/model/parse" 
    
    main()
