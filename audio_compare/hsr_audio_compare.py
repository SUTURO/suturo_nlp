import wave
import numpy as np
import rospy
from audio_common_msgs.msg import AudioData
import scipy.io.wavfile
import matplotlib.pyplot as plt
import scipy as sp

reference_sound = wave.open("/home/chris/Downloads/db2.wav", "r")
reference = reference_sound.readframes(-1)
reference = np.frombuffer(reference, np.int16).astype(np.int64)
threshold = 0.9*np.matmul(reference.astype(np.int64),reference) # 4603671836924.4
print((threshold/1e7).round(2))
# reference = reference[::-1]
N = len(reference)
# Initialize the accumulated audio data
acc_data = {"data": np.zeros(N, dtype=np.int16), "writeAt": 0} # Accumulated audio data from HSR's microphone.

# the reference sound for comparison
#with wave.open("db2.wav", "r") as reference_sound:
    #reference_data = reference_sound.readframes(-1)
maxc=0
tmax = sum(reference*reference)

def callback(data):
    global maxc
    # threshold = 1e9
    data2Append = np.frombuffer(bytes(data.data), dtype=np.int16)
    #data2Append = reference
    acc_data["writeAt"] = 0
    N = len(acc_data["data"])
    # print(N)
    M = len(data2Append)
    # print(M)
    if M > N:
        data2Append = data2Append[-N:]
    available = N-acc_data["writeAt"]
    if available >= M:
        acc_data["data"][acc_data["writeAt"]:acc_data["writeAt"]+M] = data2Append
    else:
        acc_data["data"][acc_data["writeAt"]:] = data2Append[:available]
        acc_data["data"][0:M-available] = data2Append[available:]
    acc_data["writeAt"] = (acc_data["writeAt"]+M)%N
    oldest = (acc_data["writeAt"]+1)%N
    if 0 == oldest:
        crscor = sum(acc_data["data"]*reference)
        #print(f"in if ...       crscor is:{(crscor/1e7).round(2)}")
    else:
        crscor = sum(acc_data["data"][oldest:]*reference[0:N-oldest]) + sum(acc_data["data"][:oldest]*reference[N-oldest:])
        #print(f"in else ...     crscor is:{(crscor/1e7).round(2)}")
    if crscor > maxc:
        maxc = crscor
        print("Current max", maxc, tmax, len(acc_data["data"]))
    if crscor >= threshold:
        print(f"Recognized ...  crscor is:{(crscor/1e7).round(2)}")  

'''
    # circular buffer for raw audio data
    if len(acc_data["data"]) > 16000*5: # 5 seconds
        acc_data["data"] = acc_data["data"][16000:]
        # scipy.io.wavfile.write("bla.wav", 16000, acc_data["data"])
        compare(acc_data["data"], reference_sound)

    # accumulating raw data in numpy array
    acc_data["data"] = np.concatenate([acc_data["data"], np.frombuffer(bytes(data.data), dtype=np.int16)])
    
    # print(len(acc_data["data"]))
    # print("data: ", acc_data["data"])

def compare(mic_data , reference_data):
    threshold = 100

    reference = reference[::-1]

    comp = sp.signal.fftconvolve(mic_data, reference, mode="valid")

    plt.subplots(3, 1)

    plt.subplot(3, 1, 1)
    plt.title("microphone signal")
    plt.plot(mic_data)

    plt.subplot(3, 1, 2)
    plt.title("reference signal")
    plt.plot(reference)

    plt.subplot(3, 1, 3)
    plt.title("Comparison")
    plt.plot(comp)

    print(f"The maximum value is {comp.max()}.")

    calc = np.multiply(comp.max(), np.float64(1e-9))
    if calc > threshold:
        print(f"Recognized! {calc} at {np.argmax(comp)}")
    else:
        print(f"Not recognized {calc} at {np.argmax(comp)}")
    
    plt.show()
'''

if __name__ == '__main__':

    rospy.init_node('audio_listener', anonymous=True)
    rospy.Subscriber("/audio/audio", AudioData, callback)
    rospy.spin()
