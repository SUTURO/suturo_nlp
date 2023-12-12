import requests
import speech_recognition as sr
import json
import rospy
from std_msgs.msg import String, Bool
from std_srvs.srv import SaveInfo


def main():
    #Wait for message on /startListener to continue
    #rospy.wait_for_message('/startListener', String, timeout=None)
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
        #callService = rospy.ServiceProxy('save_server', SaveInfo)
        #callService(getName(response), getDrink(response))
        # alternative code for testing
        print(f"Name: " + name + ", Drink: " + drink)
        nlpOut.publish(name + " " + drink)
    elif intent == "Receptionist" and (name == "None" or drink == "None"):
        print("Either Name or Drink was not understood.")
        nlpFeedback.publish(False)
    else:
        print("Did not undertand a Receptionist task.")
        nlpFeedback.publish(False)

    

if "__main__" == __name__:
    # Alternative code to create nlp_out topic
    nlpOut = rospy.Publisher("nlp_out", String, queue_size=16)
    nlpFeedback = rospy.Publisher("nlp_feedback", Bool, queue_size=16)
    rospy.init_node('nlp_out', anonymous=True)
    rate = rospy.Rate(1)

    # rasa Action server
    server = "http://localhost:5005/model/parse" 
    
    main()