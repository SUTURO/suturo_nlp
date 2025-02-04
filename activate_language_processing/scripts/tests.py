#!/home/siwall/venvs/whisper_venv/bin/python3.8

from nlp_mcrs_test import *
from pathlib import Path

directory1 = Path("./AudioFiles/Condition1/")
audios1 = [str(file) for file in directory1.glob("*") if file.is_file()]

directory2 = Path("./AudioFiles/Condition2/")
audios2 = [str(file) for file in directory2.glob("*") if file.is_file()]

directory3 = Path("./AudioFiles/Condition3/")
audios3 = [str(file) for file in directory3.glob("*") if file.is_file()]

directory4 = Path("./AudioFiles/Condition4/")
audios4 = [str(file) for file in directory4.glob("*") if file.is_file()]

directory5 = Path("./AudioFiles/Condition5/")
audios5 = [str(file) for file in directory5.glob("*") if file.is_file()]

directory6 = Path("./AudioFiles/Condition6/")
audios6 = [str(file) for file in directory6.glob("*") if file.is_file()]

files = [audios1, audios2, audios3, audios4, audios5, audios6]

Condition1 = 0
Condition2 = 0
Condition3 = 0
Condition4 = 0
Condition5 = 0
Condition6 = 0


# Ground truth sentences
NameAndDrink = "my name is sarah and my favorite drink is coffee."
Order = "i would like to have two steaks, fries and cola."
Hobby = "i like to play video games."

class test_mcrs:
    """
    Class for testing the nlp_mcrs.py script. The idea is to use various audio inputs, process them with mcrs and
    then match the output with a ground truth. This can be used to test and improve in the nlp_mcrs.py script. 
    """

    @staticmethod
    def run_mcrs():
        """
        This function is used to run the nlp_mcrs.py script with an audio file as input multiple times in a row.
        It captures the output and compares it to the ground truth.
        """
        NameCount = 0
        OrderCount = 0
        HobbyCount = 0
        counter = 1
        for folder in files:
            score = 0
            for audio in folder:
                transcribed_text, nlu_results = process_audio_file(audio)

                transcribed_text = transcribed_text.strip()
                transcribed_text = transcribed_text.lower()

                if "NameAndDrink" in audio:
                    groundtruth = NameAndDrink
                    if transcribed_text == groundtruth:
                        score += 1
                        NameCount += 1
                if "Order" in audio:
                    groundtruth = Order
                    if transcribed_text == groundtruth:
                        score += 1
                        OrderCount += 1
                if "Hobby" in audio:
                    groundtruth = Hobby
                    if transcribed_text == groundtruth:
                        score += 1
                        HobbyCount += 1
            
            if counter == 1:
                Condition1 = score
            elif counter == 2:
                Condition2 = score
            elif counter == 3:
                Condition3 = score
            elif counter == 4:
                Condition4 = score
            elif counter == 5:
                Condition5 = score
            elif counter == 6:
                Condition6 = score

            counter += 1

        print(Condition1)
        print(Condition2)
        print(Condition3)
        print(Condition4)
        print(Condition5)
        print(Condition6)

        print(NameCount)
        print(OrderCount)
        print(HobbyCount)
            


# Run the test
test_mcrs.run_mcrs()
