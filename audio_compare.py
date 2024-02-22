import wave
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from datetime import datetime

def main():
    print("What mode do you want to run the program in?\nnoise, comparison or library.\ncomparison promts 2 .wav files to compare. Noise promts 1 .wav and compares it with a noisy version of itself. Library promts 1 .wav and correlates it to a library of doorbells.")
    mode = input()
    if mode == "comparison" or mode == "c":
       print("Comparison mode selected.")
       userinput("c")
    elif mode == "noise" or mode == "n":
        print("Noise mode selected.") # TODO noise mode
        userinput("n")
    elif mode == "library" or mode == "l":
        print("Library mode selected.")
        userinput("l")
    elif mode == "libtest" or mode == "lt":
        print("Starting library test.")
        userinput("lt")
    else:
        print("Not a valid option. Please try again.")
        main()

def userinput(mode):
    if mode == "c":
        print("Please input a path to .wav file.")
        uaudio1 = input()

        try:
            path1 = wave.open(uaudio1, "r")
        except FileNotFoundError:
            print("File not found. Please provide a valid path to a .wav file.")
            userinput("c")
        except IsADirectoryError:
            print("Please provide a file and not a directory.")
            userinput("c")
            
        print("Next, please input a .wav file to compare to.")
        uaudio2 = input()

        try:
            path2 = wave.open(uaudio2, "r")
            compare(path1, path2)
        except FileNotFoundError:
            print("File not found. Please provide a valid path to a .wav file.")
            userinput("c")
        except IsADirectoryError:
            print("Please provide a file and not a directory.")
            userinput("c")
    
    elif mode == "n":
        print("Please input a .wav file.")
        audio = input()

        try:
            path = wave.open(audio, "r")
            compare(path, addnoise(path))
        except FileNotFoundError:
            print("File not found. Please provide a valid path to a .wav file.")
            userinput("n")
        except IsADirectoryError:
            print("Please provide a file and not a directory.")
            userinput("n")

    elif mode == "l":
        print("Please input a path to .wav file.")
        uaudio = input()

        try:
            path = wave.open(uaudio, "r")
            path = path.readframes(-1)
            path = np.frombuffer(path, np.int16)
            library(path, True)
        except FileNotFoundError:
            print("File not found. Please provide a valid path to a .wav file.")
            userinput("l")
        except IsADirectoryError:
            print("Please provide a file and not a directory.")
            userinput("l")
    
    elif mode == "lt":
        print("Please input a path to .wav file.")
        uaudio = input()

        try:
            path = wave.open(uaudio, "r")
        except FileNotFoundError:
            print("File not found. Please provide a valid path to a .wav file.")
            userinput("lt")
        except IsADirectoryError:
            print("Please provide a file and not a directory.")
            userinput("lt")
        
        print("How many runs of the test do you want to do?")
        n = int(input())

        try:
            libtest(path, n)
        except TypeError:
            print("Not a number.")
            userinput("lt")
    else:
        print("Something went wrong, please try again.")
        main()


def library(path, plot: bool):

    db1 = wave.open("/home/chris/wavs/db1.wav", "r")
    db1 = db1.readframes(-1)
    db1 = np.frombuffer(db1, np.int16)

    db2 = wave.open("/home/chris/wavs/db2.wav", "r")
    db2 = db2.readframes(-1)
    db2 = np.frombuffer(db2, np.int16)

    db3 = wave.open("/home/chris/wavs/db3.wav", "r")
    db3 = db3.readframes(-1)
    db3 = np.frombuffer(db3, np.int16)

    db4 = wave.open("/home/chris/wavs/db4.wav", "r")
    db4 = db4.readframes(-1)
    db4 = np.frombuffer(db4, np.int16)

    comp1 =  sp.signal.fftconvolve(path, db1[::-1], mode="valid")
    comp2 =  sp.signal.fftconvolve(path, db2[::-1], mode="valid")
    comp3 =  sp.signal.fftconvolve(path, db3[::-1], mode="valid")
    comp4 =  sp.signal.fftconvolve(path, db4[::-1], mode="valid")

    plt.subplots(5, 1)
    plt.tight_layout()

    plt.subplot(5, 1, 1)
    plt.title("User")
    plt.plot(path)

    plt.subplot(5, 1, 2)
    plt.title(f"User x db1: {np.multiply(comp1.max(), np.float64(1e-11))} at {np.argmax(comp1)}")
    plt.plot(comp1)

    plt.subplot(5, 1, 3)
    plt.title(f"User x db2: {np.multiply(comp2.max(), np.float64(1e-11))} at {np.argmax(comp2)}")
    plt.plot(comp2)

    plt.subplot(5, 1, 4)
    plt.title(f"User x db3: {np.multiply(comp3.max(), np.float64(1e-11))} at {np.argmax(comp3)}")
    plt.plot(comp3)

    plt.subplot(5, 1, 5)
    plt.title(f"User x db4: {np.multiply(comp4.max(), np.float64(1e-11))} at {np.argmax(comp4)}")
    plt.plot(comp4)

    if plot:
        plt.show()
    else:
        print(f"""User x db1: {np.multiply(comp1.max(), np.float64(1e-11))} at {np.argmax(comp1)},
                  \nUser x db2: {np.multiply(comp2.max(), np.float64(1e-11))} at {np.argmax(comp2)},
                  \nUser x db3: {np.multiply(comp3.max(), np.float64(1e-11))} at {np.argmax(comp3)},
                  \nUser x db4: {np.multiply(comp4.max(), np.float64(1e-11))} at {np.argmax(comp4)}""")
        
        time = datetime.now().strftime('%H_%M_%S_%f')
        plt.savefig(f"/home/chris/Pictures/libtest_{time[:-3]}.png")

def libtest(path, n: int):
    data = path.readframes(-1)
    data = np.frombuffer(data, np.int16)

    while n > 0:
        prepend_zeros = np.zeros(np.random.randint(0, len(data)))
        append_zeros = np.zeros(np.random.randint(0, len(data)))

        pdata = np.concatenate((prepend_zeros, data, append_zeros))
        print(f"Prepend: {len(prepend_zeros)}, append: {len(append_zeros)}, Data: {len(pdata)}")

        library(pdata, False)
        n = n-1
        print(f"{n} runs left")

    print("Tests done.")


def addnoise(data):
    np.random.normal(0, 1, 1000) + data.readframes(-1)


    
def compare(a1, a2):
    threshold = 5

    sig1 = a1.readframes(-1)
    sig1 = np.frombuffer(sig1, np.int16)
    #pad = np.zeros((10000,),np.int16)
    #sig1 = np.concatenate((pad,sig1))

    sig2 = a2.readframes(-1)
    sig2 = np.frombuffer(sig2, np.int16)
    sig1l = sig1.tolist()
    sig2l = sig2.tolist()
    sig2 = sig2[::-1]

    comp = sp.signal.fftconvolve(sig1, sig2, mode="valid")
    # comp = sp.signal.correlate(sig1, sig2)
    # comp = sp.signal.coherence(sig1, sig2)
    new =  []
    sig1l = sig1.tolist()
    sig2l = sig2.tolist()
    #for k in range(200):
    #    new.append(sum([x*y for x,y in zip(sig1l[k:],sig2l)]))

    plt.subplots(4, 1)

    plt.subplot(4, 1, 1)
    plt.title("Signal 1")
    plt.plot(sig1)

    plt.subplot(4, 1, 2)
    plt.title("Signal 2")
    plt.plot(sig2)

    plt.subplot(4, 1, 3)
    plt.title("Comparison")
    plt.plot(comp)

    #plt.subplot(4, 1, 4)
    #plt.title("Manual Calculation < 200")
    #plt.plot(new)

    calc = np.multiply(comp.max(), np.float64(1e-11))
    if calc > threshold:
        print(f"Recognized! {calc} at {np.argmax(comp)}")
    else:
        print(f"Not recognized {calc} at {np.argmax(comp)}")
    
    plt.show()


if __name__ ==  "__main__":
    main()