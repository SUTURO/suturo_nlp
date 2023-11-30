import requests
import whisper
import speech_recognition as sr
import json
import rospy
from std_msgs.msg import String

'''
For the program to run, you need to have roscore and the rasa-server running.
The recorder will start on receiving any String message on the /startListener topic.
'''
def main():
    #Wait for message on /startListener to continue
    rospy.wait_for_message('/startListener', String, timeout=None)
    record()

'''
This function records from the microphone, sends it to whisper and to the Rasa server
'''
def record():  
    # Record with speech recognizer for as long as someone is talking
    r = sr.Recognizer()
    r.pause_threshold = 2
    with sr.Microphone() as source:
        print("Say something!")
        r.adjust_for_ambient_noise(source, 1)
        audio = r.listen(source)

    # Write recorder audio to file
    with open("temp_file", "wb") as file:
        file.write(audio.get_raw_data())

    # load small english
    #model = whisper.load_model("small.en")

    # load audio and transcribe with whisper
    #result = model.transcribe(audio, fp16=False).get("text", "").lower()    
    result = r.recognize_whisper(audio, language="english")

    # send result to RASA
    ans = requests.post(server, data=bytes(json.dumps({"text": result}), "utf-8"))

    # load answer from RASA 
    response = json.loads(ans.text)   

    # change response format
    response = {"text": response.get("text", ), "intent": response.get("intent", {}).get("name"), 
                "entities": set([(x.get("entity"), x.get("value")) for x in response.get("entities", [])])}
    
    # publish response on nlp_out
    pub.publish(str(response))

if "__main__" == __name__:
    # Initiate startListener topic. On message, call record()
    #rospy.init_node('startListener', anonymous=True)

    # Create nlp_out topic
    pub = rospy.Publisher("nlp_out", String, queue_size=16)

    rospy.init_node('nlp_out', anonymous=True)
    rate = rospy.Rate(1)

    # rasa Action server
    server = "http://localhost:5005/model/parse" 
    
    main()