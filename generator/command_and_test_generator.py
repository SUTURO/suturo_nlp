# New and improved sentence generator for the RoboCup organizers. Watch and learn, noobs.

import argparse
import random
import json
import os
import sys
import copy
import nltk

nltk.download('punkt')

# We may want to run this script from an arbitrary location, let's not depend on the current working directory to locate our resources.
ownDir = os.path.dirname(os.path.abspath(__file__))

# TODO: get rid of these placeholder intents from our rules and stories
preamble = '''
- intent: ParkingArms
  examples: |
    - Please place your arms in a relaxed position.
    - Please park your arms.
    - Kindly rest your arms at your sides.
    - Can you put your arms down and let them rest?
    - Can you put your hands near your body?
    - Kindly rest your hands at your sides.

- intent: greet
  examples: |
    - hey
    - hello
    - hi
    - hello there
    - good morning
    - good evening
    - moin
    - hey there
    - let's go
    - hey dude
    - goodmorning
    - goodevening
    - good afternoon

- intent: goodbye
  examples: |
    - cu
    - good by
    - cee you later
    - good night
    - bye
    - goodbye
    - have a nice day
    - see you around
    - bye bye
    - see you later

- intent: affirm
  examples: |
    - yes
    - y
    - indeed
    - of course
    - that sounds good
    - correct

- intent: deny
  examples: |
    - no
    - n
    - never
    - I don't think so
    - don't like that
    - no way
    - not really

- intent: mood_great
  examples: |
    - perfect
    - great
    - amazing
    - feeling like a king
    - wonderful
    - I am feeling very good
    - I am great
    - I am amazing
    - I am going to save the world
    - super stoked
    - extremely good
    - so so perfect
    - so good
    - so perfect

- intent: mood_unhappy
  examples: |
    - my day was horrible
    - I am sad
    - I don't feel very well
    - I am disappointed
    - super sad
    - I'm so sad
    - sad
    - very sad
    - unhappy
    - not good
    - not very good
    - extremly sad
    - so saad
    - so sad

- intent: bot_challenge
  examples: |
    - are you a bot?
    - are you a human?
    - am I talking to a bot?
    - am I talking to a human?

'''

def loadSharp(infile):
    def _findCommentStart(l): # Kinda silly to use # for comment and ### for situations in cat3, but if them's the format ...
        retq = -1
        for k, c in enumerate(l):
            if '#' == c:
                if -1 == retq: # Encountered a # without being triggered ...
                    retq = k # ... get triggered
                elif 2 == k - retq: # we are triggered, and encountered a third consecutive # ...
                    retq = -1 # ... untrigger, this is a situation marker; NOTE: if we are triggered and encounter a second #, nothing happens: we just stay triggered
            elif -1 != retq: # We are triggered and encountered anything other than a # ...
                break # ... can untrigger, we have found a comment start
        return retq
    def _removeComment(l): # Lets get rid of comments everywhere, shall we. A comment is whatever follows an occurrence of a lone or paired, but not trebled, #
        idx = _findCommentStart(l)
        if -1 != idx:
            return l[:idx]
        return l
    # We are being properly permissive with blank spaces, including tabs: an arbitrary number of them can be removed from start and end. Also, ignore empty lines when assembling the final result.
    return [y for y in [_removeComment(x.strip()) for x in open(os.path.join(ownDir, infile), 'r').readlines()] if (0 < len(y))]

# Somewhat inflexible to only have items, locations, names. What if we need more entity types?
def loadOntology(ontologyLoader):
    retq = {}
    for entityType, fileEntities, fileClassLabels, msgName in ontologyLoader:
        retq[entityType] = {'entities': loadSharp(fileEntities), # Isn't it much nicer to have a function you can call when you need to do the same thing over and over again?
                            'classes': None}
        if 1 > len(retq[entityType]['entities']):
            print("Do not have any %s defined in the file!" % msgName)
            sys.exit(1)
        if fileClassLabels is not None:
            retq[entityType]['classes'] = loadSharp(fileClassLabels)
            if len(retq[entityType]['entities']) != len(retq[entityType]['classes']):
                print("Different number of %s than the number of categories for them." % msgName)
                sys.exit(1)
        else: # We will treat *_CATEGORY as its own class of entities. However, in the corresponding file, entities appear several times.
            retq[entityType]['entities'] = list(set(retq[entityType]['entities']))
    return retq

ontologyLoader = []
ontologyLoader.append(("NaturalPerson", "naturalPersons.txt", None, "NaturalPersons"))
ontologyLoader.append(("drink", "drinks.txt", None, "drinks"))
ontologyLoader.append(("food", "foods.txt", None, "foods"))
ontologyLoader.append(("PhysicalArtifact", "physicalArtifacts.txt", None, "PhysicalArtifacts"))
ontologyLoader.append(("PhysicalPlace", "physicalPlaces.txt", None, "PhysicalPlaces"))

#ontologyLoader.append(('LOCATION', 'locations.txt', 'location_categories.txt', 'locations'))
#ontologyLoader.append(('ITEM', 'items.txt', 'item_categories.txt', 'items'))
#ontologyLoader.append(('LOCATION_CATEGORY', 'location_categories.txt', None, 'locations'))
#ontologyLoader.append(('ITEM_CATEGORY', 'item_categories.txt', None, 'items'))
#ontologyLoader.append(('NAME', 'names.txt', None, 'names'))
#ontologyLoader.append(('TASK', 'tasks.txt', None, 'tasks'))

#ontologyLoader.append(('object', 'objects.txt', 'object_categories.txt', 'objects'))
#ontologyLoader.append(('object_category', 'object_categories.txt', None, 'object_categories'))
#ontologyLoader.append(('location', 'locations.txt', 'location_categories.txt', 'locations'))
#ontologyLoader.append(('location_category', 'location_categories.txt', None, 'location_categories'))
#ontologyLoader.append(('room', 'rooms.txt', None, 'rooms'))
#ontologyLoader.append(('colours', 'colours.txt', None, 'colours'))
#ontologyLoader.append(('attributes', 'attributes.txt', None, 'attributes'))
#ontologyLoader.append(('gestures', 'gestures.txt', None, 'gestures'))
#ontologyLoader.append(('number', 'numbers.txt', None, 'numbers'))
#ontologyLoader.append(('person_name', 'person_names.txt', None, 'person_names'))
#ontologyLoader.append(('person', 'persons.txt', None, 'persons'))

ontology = loadOntology(ontologyLoader)

templatesRaw = loadSharp('templates.txt')
templates = {}
for t in templatesRaw:
    idx = t.find(':')
    intent = t[:idx].strip()
    sentence = t[idx+1:].strip()
    if intent not in templates:
        templates[intent] = []
    templates[intent].append(sentence)

punctuation = set([',', ':', '.', ';', '"', '\'', '?', '!'])

# We actually may want to generate MANY sentences at once for both training and testing purposes. One by one is silly.
def generateN(N, Ntrials, templates, ontology):
    creations = {}
    for intent in sorted(list(templates.keys())): # TODO: may want to force generation of O(N) examples per template
        intentTemplates = templates[intent]
        creations[intent] = {}
        k = 0
        while (N > len(creations[intent])) and (k < Ntrials):
            k += 1
            template = random.choice(intentTemplates)
            tokens = nltk.word_tokenize(template)
            spec = []
            for token in tokens:
                value = str(token)
                annotation = None
                if '|' in token:
                    annspec = token.split('|')
                    annotation = {"entity": annspec[0], "role": annspec[1]}
                    if 2 < len(annspec):
                        annotation["group"] = annspec[2]
                    if annspec[0] in ontology:
                        value = random.choice(ontology[annspec[0]]['entities'])
                    else:
                        value = annspec[0] # TODO: this imposes a limit: an out-of-onto token must be a single "word" according to nltk.word_tokenize.
                        # This means: OK: "yourself", "me". Not ok: "cup of coffee".
                spec.append([value, annotation])
            creations[intent][specToText(spec)] = spec
    return {intent: [(k, creations[intent][k]) for k in sorted(list(creations[intent].keys()))] for intent in sorted(list(creations.keys()))}

def specToText(s):
    retq = ""
    for token, _ in s:
        separator = ''
        if token not in punctuation:
            separator = ' '
        retq += separator + token
    return retq.strip()

def specToRasaTrain(s):
    retq = ""
    for token, annotation in s:
        separator = ''
        if token not in punctuation:
            separator = ' '
        piece = str(token)
        if annotation is not None:
            piece = '[%s]%s' % (token, json.dumps(annotation))
        retq += separator + piece
    return retq.strip()

def specToEntities(s):
    retq = []
    for token, annotation in s:
        if annotation is None:
            continue
        entityType, role, group = annotation.get("entity"), annotation.get("role"), annotation.get("group")
        aux = [entityType, role, token]
        if group is not None:
            aux.append(group)
        retq.append(aux)
    return retq

def writeNL(creations, outfileName):
    with open(outfileName, 'w') as outfile:
        for intent in sorted(list(creations.keys())):
            specs = creations[intent]
            for text, _ in specs:
                _ = outfile.write("%s\n" % text)

def writeRasaNLU(creations, outfileNLUName, outfileDomainName, ontology):
    with open(outfileNLUName, 'w') as outfile:
        _ = outfile.write('version: "3.1"\n\nnlu:\n')
        _ = outfile.write("%s\n" % preamble)
        for intent in sorted(list(creations.keys())):
            specs = creations[intent]
            _ = outfile.write('- intent: %s\n  examples: |\n' % intent)
            for _, s in specs:
                _ = outfile.write('    - %s\n' % specToRasaTrain(s))
            _ = outfile.write('\n')
        for e in sorted(list(ontology.keys())):
            _ = outfile.write('- lookup: %s\n  examples: |\n' % e)
            for x in sorted(list(set(ontology[e]['entities']))):
                _ = outfile.write('    - %s\n' % x)
            _ = outfile.write('\n')
    with open(outfileDomainName, 'w') as outfile:
        _ = outfile.write('version: "3.1"\n\nintents:\n')
        for intent in sorted(list(creations.keys())):
            _ = outfile.write('  - %s\n' % intent)
        _ = outfile.write('\n\nentities:\n')
        for e in sorted(list(ontology.keys())):
            _ = outfile.write(' - %s\n' % e)
        _ = outfile.write('\nsession_config:\n  session_expiration_time: 60\n  carry_over_slots_to_new_session: true\n')

def writeRasaTesting(creations, outfileName):
    with open(outfileName, 'w') as outfile:
        tests = []
        for intent in sorted(list(creations.keys())):
            specs = creations[intent]
            for text, s in specs:
                aux = {"text": text,
                       "response": {"intent": intent,
                                    "entities": specToEntities(s)}}
                tests.append("\t%s" % (json.dumps(aux)))
        _ = outfile.write('[\n%s\n]\n' % ',\n'.join(tests))

def main():
    # TODO: provide an arbitrary path for output files.
    parser = argparse.ArgumentParser(prog='command_and_test_generator', description='Generate commands, training, and testing files for NLU commands', epilog='Text at the bottom of help')
    parser.add_argument('-Ntrain', '--numberTrainingExamples', default="20", help='Number of training examples to generate per intent. Will be forced to 20 if lower.')
    parser.add_argument('-Ntest', '--numberTestingExamples', default="20", help='Number of training examples to generate per intent. Will be forced to 20 if lower.')
    parser.add_argument('-Ntries', '--numberTrials', default="400", help='Number of trials for random generation per intent. Forced to 10*(Ntrain+Ntest) if lower.')
    # Forget about generating according to their category files. Internally, our team is splitting sentences so we need to generate train/test data for intents.
    #parser.add_argument('-c', '--category', default="2", help='Category of templates to generate for. May be 1 or 2 (will be forced to 2 otherwise). TODO: add support for cat 3.')
    arguments = parser.parse_args()
    Ntrain = int(arguments.numberTrainingExamples)
    if 20 > Ntrain:
        Ntrain = 20
    Ntest = int(arguments.numberTrainingExamples)
    if 20 > Ntest:
        Ntest = 20
    Ntrials = int(arguments.numberTrials)
    if 10*(Ntrain + Ntest) > Ntrials:
        Ntrials = 10*(Ntrain + Ntest)
    trainSpecs = generateN(Ntest, Ntrials, templates, ontology)
    testSpecs = generateN(Ntest, Ntrials, templates, ontology)
    writeNL(trainSpecs, 'generated_training.txt')
    writeNL(testSpecs, 'generated_testing.txt')
    writeRasaNLU(trainSpecs, 'nlu.yml', 'domain.yml', ontology)
    writeRasaTesting(testSpecs, 'tests.json')

if "__main__" == __name__:
    main()

