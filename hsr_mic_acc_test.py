import rospy
from audio_common_msgs.msg import AudioData
from std_msgs.msg import String, Bool
import numpy
import whisper

import speech_recognition as sr

# to save the uint8 data we get from hrs's microphone as whisper readable int16
acc_data = numpy.array([], dtype=numpy.int16)
result_list = []

model = whisper.load_model("small.en")

def main():
    # after capture_wave was launched on the hsr we can subscribe and get the audio data
    rospy.Subscriber('/audio/audio', AudioData, accumulate)
    rospy.spin()

def accumulate(data):

    # TODO needs to be global, don't know why
    global acc_data

    # recording get's longer
    acc_data = numpy.concatenate([acc_data, numpy.frombuffer(bytes(data.data), dtype=numpy.int16)])
    
    # just to check how the array fills up when there is silence or someone is talking
    rospy.loginfo("Now I'm thaaaat big: %d elements", len(acc_data))

    #print(acc_data)

    # has to be float for whisper
    float_data = acc_data.astype(numpy.float32, order='C') / 32768.0

    #print(float_data)

    # just for testing TODO smarter solution, queue?
    if len(acc_data) >= 20000:

        # this should work

        result = model.transcribe(float_data, fp16=False).get("text", "").lower()
        result_list.append(result) # so we just have to save the strings
        
        print(result_list)

        # perhaps we should do something like that until we use a queue but may produce gaps in the recording
        acc_data = numpy.array([], dtype=numpy.uint16)
        
        # if you don't want to loop forever, uncomment
        # rospy.signal_shutdown("the end")


        # this also works, but with speech recognition
        # TODO so perhaps we can still use the end detection?

        # r = sr.Recognizer()
        # audio_source = sr.AudioData(acc_data, 16000, 2)
        # result = r.recognize_whisper(audio_source, language="german")
        # print(result)
        # rospy.signal_shutdown("the end")

if "__main__" == __name__:
    rospy.init_node('nlp_out', anonymous=True)
    rate = rospy.Rate(1)
    
    main()