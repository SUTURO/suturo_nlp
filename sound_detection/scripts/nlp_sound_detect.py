#!/usr/bin/env python3

import numpy as np
import rospy
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData
import scipy.io.wavfile
import scipy as sp
import rospkg
import rospkg
# import time

# get the path to our package
rospack = rospkg.RosPack()
package_path = rospack.get_path('sound_detection')

# construct full path to reference file
reference_file_path = package_path + "/scripts/reference_16K.wav"

# we don't care about the sample rate value, we just save the audio data of the reference sound
_, reference_sound = scipy.io.wavfile.read(reference_file_path)
reference = reference_sound

# Initialize the numpy array to store the audio data
acc_data = {"data": np.zeros(len(reference), dtype=np.int16), "writeAt": 0}

# Initialize the compare frequency
compare_frequecy = 0

def callback_fft(data):

    global compare_frequecy
    compare_frequecy += 1
    
    new_data = np.frombuffer(bytes(data.data), dtype=np.int16)

    acc_data["data"] = np.concatenate([acc_data["data"][len(new_data):], new_data])
    
    # Compare only every 100 samples to save computation time 
    if compare_frequecy % 100 == 0:
        compare(acc_data["data"])

def compare(mic_data, threshold=None):

    if threshold is None:
        threshold = 100
    
    # s = time.perf_counter() # for performance measurement
    comp = sp.signal.fftconvolve(mic_data, reference, mode="valid")

    calc = np.multiply(comp.max(), np.float64(1e-9))
    # print(time.perf_counter()-s) # for performance measurement

    if calc > threshold:
        nlpOut.publish(f"<DOORBELL>")
        rospy.signal_shutdown('Doorbell detected')


if __name__ == '__main__':

    # Initialize ros node
    rospy.init_node('audio_listener', anonymous=True)    
    # Publisher for the nlp_out topic
    nlpOut = rospy.Publisher("nlp_out", String, queue_size=16)
    rospy.Subscriber("/audio/audio", AudioData, callback_fft)
    rospy.spin()