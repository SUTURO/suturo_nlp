Prerequisites:

Assumes python3 and rasa are already installed. Open a terminal, and type in the following commands, one by one.

sudo pip install nltk
python3
import nltk
nltk.download('punkt')
exit()

Once the above is done, here is the generate-train-test procedure:

Open a terminal and go to this folder. Then run the following commands, one by one:

python3 ./command_and_test_generator.py
cp ./nlu.yml ../data/nlu.yml
cd ../
rasa train
rasa run --enable-api

Open a different terminal and go to this folder. Then run
python3 ./testing.py

The results are in errors.log.

