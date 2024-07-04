#!/usr/bin/env python3

import argparse
import speech_recognition as sr
import logging
import pygame
import threading
import time

# recomended command if you don't want to get these terrible ALSA warnings
# python3 activate_noisy_input.py 2>/dev/null/ 

def test_with_noise(pause_threshold, energy_threshold):
    '''
plays some annoying sounds while you can test the speech recognition
*pause_threshold* silence until break
*energy_threshold* sensitivity
    '''

    r = sr.Recognizer()
    r.pause_threshold = pause_threshold

    # Higher values mean that it will be less sensitive, which is useful if you are in a loud room.
    # There is no one-size-fits-all value, but good values typically range from 50 to 4000.
    r.energy_threshold = energy_threshold

    while True:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, 2)  # ok we obviously need this.. but I think it doesn't make sense
            print("Hello dear guest, please tell me your name and your favourite drink.")
            audio = r.listen(source)

        # Write recorder audio to file
        with open("temp_file", "wb") as file:
            file.write(audio.get_raw_data())

        # Use sr Whisper integration
        result = r.recognize_whisper(audio, language="english")
        result = result.strip()
        print(f"The text I recognized is: {result}\n")
        # TODO: could be nice to save responses from the user for training statistics

        # exit the programm, could happen accidently
        if result.lower() in ["bye"]:
            print("Bye.")
            break

# to play the annoying sound file
def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()


if __name__ == "__main__":

    # to manipulate pause and energy via args, just use the -h flag
    parser = argparse.ArgumentParser(description='Adjust pause between sentences and .')
    parser.add_argument('-p', '--pause_threshold', type=float, default=2, help='Adjust pause between sentences.')
    parser.add_argument('-e', '--energy_threshold', type=int, default=4000,
                        help='Manipulate energy threshold value (good values typically range from 50 to 4000).')

    args = parser.parse_args()

    print(f"## Please keep a gap of {args.pause_threshold} seconds between tests. ##\n")

    # start another thread for playing the audio file
    audio_thread = threading.Thread(target=play_audio, args=("./convention_hall_ambience_sounds.wav",))
    audio_thread.start()

    # some time, so the recording can start properly
    time.sleep(1)

    test_with_noise(args.pause_threshold, args.energy_threshold)
