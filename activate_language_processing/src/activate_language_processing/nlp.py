import json
import requests
import spacy

import subprocess
import sys

def install_spacy_required_packages():
    packages = ['en_core_web_sm']
    for package_name in packages:
        if not spacy.util.is_package(package_name):
            subprocess.check_call([sys.executable, "-m", "spacy", "download", package_name])

install_spacy_required_packages()

placeholderWords = {"her", "him", "it", "them", "there"}

def parseIntent(text, context):
    """
Use RASA to parse a simple sentence (one intent).
    """
    req = {"text": text}
    r = requests.post(context["rasaURI"],  data=bytes(json.dumps(req), "utf-8"))
    response = json.loads(r.text)
    retq = {"sentence": text, "intent": response['intent']['name'], "entities": {}}
    for k,e in enumerate(response["entities"]):
        retq["entities"][k] = [e.get("role", "UndefinedRole"), e.get("value", "UnparsedEntity"), e.get("group", 0)]
    return retq

def degroup(parses):
    """
Convert a parse that may have groups (sets of entities to act on in the same way in parallel) into a list
of parses.
    """
    retq=[]
    for e in parses:
        intent=e["intent"]
        entities=e["entities"]
        groups={0:{}}
        for k, ed in entities.items():
            role,value,group=ed
            if group not in groups:
                groups[group]={}
            if role not in groups[group]:
                groups[group][role]=set()
            groups[group][role].add((k, value))
        for k in sorted(groups.keys()):
            eds = {}
            for r,vs in groups[k].items():
                for kk,v in vs:
                    eds[kk]= [r,v,0]
            retq.append({"sentence":e["sentence"], "intent":intent, "entities": eds})
    return retq

def getSubtree(tok):
    """
Return the subtree of a token, but stop at dependent verbs.
This allows splitting a text into sentences.
    """
    inText=[(tok.idx, tok)]
    todo=list(tok.children)
    next = []
    while todo:
        cr=todo.pop()
        if ("VERB" == cr.pos_):
            next.append(cr)
        else:
            inText.append((cr.idx,cr))
            todo = todo + list(cr.children)
    toks = [str(x[1]) for x in sorted(inText,key=lambda x:x[0])]
    return next, ' '.join(toks)

def splitIntents(text, context):
    doc=context["nlp"](text)
    intentUtterances=[]
    for s in doc.sents:
        todo=[s.root]
        while todo:
            cr = todo.pop()
            next, text = getSubtree(cr)
            todo = todo+next
            intentUtterances.append(text)
    return intentUtterances

def guessRoles(parses, context, needsGuessFn):
    def _te2de(entities):
        retq = {}
        for k,v in entities.items():
            role,value,_=v
            if role not in retq:
                retq[role]=set()
            retq[role].add(value)
        return retq
    def _de2te(entities):
        retq={}
        j=0
        for k,vs in entities.items():
            for v in vs:
                retq[j] = (k,v,0)
                j += 1
        return retq
    roleMap={}
    retq=[]
    for e in parses:
        intent=e["intent"]
        entities=_te2de(e["entities"])
        for role, fillers in entities.items():
            if needsGuessFn(fillers):
                for guessedRole in {role}.union(context["role2Roles"].get(role,[])):
                    if guessedRole in roleMap:
                        entities[role]=roleMap[guessedRole]
                        break
            elif 0<len(fillers):
                roleMap[role]=fillers
        retq.append({"sentence":e["sentence"],"intent":intent,"entities":_de2te(entities)})
    return retq

def semanticLabelling(text, context):
    intentUtterances=splitIntents(text, context)
    parsedIntents=degroup([parseIntent(x, context) for x in intentUtterances])
    parsedIntents=guessRoles(parsedIntents, context, lambda x: 0!=len(x.intersection(placeholderWords)))
    for k,e in enumerate(parsedIntents):
        if (0==len(e["entities"])) and (k < len(parsedIntents)-1):
            j=0
            for role, value, group in parsedIntents[k+1]["entities"].values():
                 if role in context["intent2Roles[e["intent"]]:
                     e["entities"][j] = (role,value,group)
                     j+=1
    return parsedIntents
