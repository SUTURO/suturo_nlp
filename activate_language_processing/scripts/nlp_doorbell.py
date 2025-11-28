#!/usr/bin/env python3

from argparse import ArgumentParser
import numpy as np
import rospy
from audio_common_msgs.msg import AudioData
import scipy.io.wavfile
import scipy as sp
import time

# Every time new data arrives form our microphone execute this function
def callback_fft(context, data):

    # Convert data into int16 np array
    new_data = np.frombuffer(bytes(data.data), dtype=np.int16)
    # Remove old samples and append newest
    context["data"] = np.concatenate([context["data"][len(new_data):], new_data])
    
    context["divider"] += 1
    # Compare only every 100 samples to save computation time 
    if 100 <= context["divider"]:
        context["divider"] = 0
        compare(acc_data["data"], context["threshold"], context["pub"])

# Compare the audio with the reference sound from the .wav file using FFT
# Convolve audio sample with the reference
# If we found a peak that is bigger than the threshold we found most likely our bell
def compare(mic_data, threshold, pub):
    # s = time.perf_counter() # for performance measurement
    comp = sp.signal.fftconvolve(mic_data, reference, mode="valid")

    calc = np.multiply(comp.max(), np.float64(1e-9))
    # print(time.perf_counter()-s) # for performance measurement

    if calc > threshold:
        pub.publish("Doorbell detected!")

if __name__ == '__main__':
    rospy.init_node('doorbell_detection', anonymous=True)
    rospy.loginfo("[ALP]: Doorbell detection node initialized")

    parser = ArgumentParser(prog='nlp_doorbell', description='Publishes a string message to a particular notification topic when a doorbell has rung.')
    parser.add_argument('-o', '--outputTopic', default="bell_rang", help='Name of topic to send notifications to. Default: bell_rang')
    parser.add_argument('-r', '--referenceWAV', default="db2_16K.wav", help='File path (recommended: absolute, not relative path) to a WAV file storing the reference doorbell. This WAV MUST be mono (not stereo!), 16ksps, 16bit/sample.')
    parser.add_argument('-t', '--threshold', default='10.0', help='Threshold to declare the input sound similar enough to the reference doorbell. Default: 10.0')
    args, unknown = parser.parse_known_args(rospy.myargv()[1:])

    # The reference audio file
    referenceWAVFile = str(args.referenceWAV)
    threshold = float(args.threshold)
    # Topic to publish
    topic = args.outputTopic

    # Publish String messages to topic
    pub = rospy.Publisher(topic, String, queue_size=1)

    _, reference = scipy.io.wavfile.read(referenceWAVFile)
    context = {"data": np.zeros(len(reference), dtype=np.int16), "reference": reference, "threshold": threshold, "divider": 0, "pub": pub}

    # Subscriber for microphone data
    rospy.Subscriber("/audio/audio", AudioData, lambda data: callback_fft(context, data))
    rospy.loginfo("[ALP]: Doorbell detection node started")

    rospy.spin()
