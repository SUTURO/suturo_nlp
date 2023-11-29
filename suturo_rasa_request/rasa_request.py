#!/usr/bin/python3
#coding=utf-8
import requests     # HTTP requests
import whisper      # voice recognition
import json         # JSON format export and import
import subprocess   # run system commands
import time	        # to create timestamp
 
def main():

    input("press enter to start recording")  # to run the script

    # timestamp for naming the audio file
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # creating a filename for the recording
    filename = f"./recordings/voice_recording_{timestamp}.m4a"

    # run ffmpeg command in shell to start audio recording
    start_recording_cmd = f"ffmpeg -f alsa -i hw:0 {filename}"
    subprocess.run(start_recording_cmd, shell=True)

    server = "http://localhost:5005/model/parse"    # run http rasa server

    model = whisper.load_model("small.en")          # load small english model

    # load audio file and recognize text with whisper
    result = model.transcribe(filename, fp16=False).get("text", "").lower()    
    
    print("\n" + result + "\n")     # test output to check if recognition worked well

    # send json to RASA and save the response
    ans = requests.post(server, data=bytes(json.dumps({"text": result}), "utf-8"))

    response = json.loads(ans.text) # load response as json

    # transform response to a useful and readable JSON file
    response = {"text": response.get("text", ), "intent": response.get("intent", {}).get("name"), 
                "entities": set([(x.get("entity"), x.get("value")) for x in response.get("entities", [])])}

    print(response)     # Output

if "__main__" == __name__:
    main()
    
    
    
    
    
