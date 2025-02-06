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
conjDeps = {"conj", "dep"}
auxDeps = {"ccomp"}
attrDeps = {"acl", "amod", "relcl"}
numDeps = {"nummod"}
actAttrPOS = {"VERB"}
propAttrPOS = {"ADJ", "ADV"}
numAttrPOS = {"NUM"}
roleForbiddenDeps = attrDeps.union(numDeps)

def inRange(idx, idxS, idxE):
    return (idxS<=idx) and (idx<idxE)

def rainDance(text):
    """
Perform rain dances to hopefully appease the Neural Network gods
so as to get a good parse out of a text.

In boring speak, a few preprocessing steps that MIGHT steer spacy
away from some dumb failures.
    """
    # Avoid a then-clause -advcls-> previous clause, should instead have previous clause -dep|conj-> then-clause
    text = text.replace(", then", " then").replace("'d", " would").replace(",", " ,").replace("'ll", " will")
    return text

def getAttributes(idxS, idxE, idx2Tok, deps, poss):
    retq = []
    for idx, tok in idx2Tok.items():
       if inRange(idx, idxS, idxE):
            for c in tok.children:
                if (not inRange(c.idx, idxS, idxE)) and (c.dep_ in deps) and (c.pos_ in poss):
                    _, text, _ = getSubtree(c)
                    retq.append(text)
    return tuple(retq)

def subtreeDep(idxS, idxE, idx2Tok):
    inSpan = set()
    idx2Dep = {}
    for idx, tok in idx2Tok.items():
        if inRange(idx, idxS, idxE):
            inSpan.add(idx)
            idx2Dep[idx] = (tok.head.idx, tok.dep_)
    for idx, dep in idx2Dep.items():
        hIdx, dep_ = dep
        if hIdx not in inSpan:
            return dep_
    return None
            
def parseIntent(cspec, context):
    """
Use RASA to parse a simple sentence (one intent).
    """
    text = cspec["text"]
    sStart = cspec["start"]
    idx2Tok = cspec["idx2Tok"]
    req = {"text": text}
    r = requests.post(context["rasaURI"],  data=bytes(json.dumps(req), "utf-8"))
    response = json.loads(r.text)
    retq = {"sentence": text, "intent": response['intent']['name'], "entities": {}}
    for k,e in enumerate(response["entities"]):
        #print("Entity", e, sStart)
        eStart = e.get("start", 0)+sStart
        eEnd = e.get("end", 0)+sStart
        if subtreeDep(eStart, eEnd, idx2Tok) in roleForbiddenDeps:
            continue
        retq["entities"][k] = {"idx": k, "role": e.get("role", "UndefinedRole"), "value": e.get("value", "UnparsedEntity"), "group": int(e.get("group", 0)), "entity": e.get("entity", "owl:Thing"), "propertyAttribute": getAttributes(eStart, eEnd, idx2Tok, attrDeps, propAttrPOS), "actionAttribute": getAttributes(eStart, eEnd, idx2Tok, attrDeps, actAttrPOS), "numberAttribute": getAttributes(eStart, eEnd, idx2Tok, numDeps, numAttrPOS)}
    return retq

def degroup(parses):
    """
Convert a parse that may have groups (sets of entities to act on in the same way in parallel) into a list
of parses.
    """
    print(parses)
    retq=[]
    for e in parses:
        intent=e["intent"]
        entities=e["entities"]
        groups={0:{}}
        for k, ed in entities.items():
            role = ed["role"]
            group = ed["group"]
            if group not in groups:
                groups[group]={}
            if role not in groups[group]:
                groups[group][role]=[]
            groups[group][role].append(ed)
        for k in sorted(groups.keys()):
            eds = {}
            for r,vs in groups[k].items():
                for v in vs:
                    eds[v["idx"]]= v.copy()
                    eds[v["idx"]]["group"] = 0
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
    idx2Tok = {tok.idx: tok}
    excluded = set()
    for c in tok.children:
        if (("VERB" == c.pos_) and (c.dep_ in conjDeps)) or (("AUX" == c.pos_ ) and ((c.dep_ in conjDeps) or (c.dep_ in auxDeps)) and ("be" == c.lemma_)):
        #if (("VERB" == c.pos_) and (c.dep_ in conjDeps)):
            next.append(c)
            excluded.add(c.idx)
    while todo:
        cr=todo.pop()
        if cr.idx not in excluded:
            inText.append((cr.idx,cr))
            idx2Tok[cr.idx] = cr
            todo = todo + list(cr.children)
    toks = [str(x[1]) for x in sorted(inText,key=lambda x:x[0])]
    return next, ' '.join(toks), idx2Tok

def splitIntents(text, context):
    doc=context["nlp"](text)
    intentUtterances=[]
    for s in doc.sents:
        todo=[s.root]
        while todo:
            cr = todo.pop()
            next, text, idx2Tok = getSubtree(cr)
            todo = todo+next
            intentUtterances.append({"text": text, "start": min(idx2Tok.keys()), "idx2Tok": idx2Tok})
    return intentUtterances

def guessRoles(parses, context, needsGuessFn):
    def _te2de(entities):
        retq = {}
        for k,v in entities.items():
            role=v["role"]
            if role not in retq:
                retq[role]=[]
            retq[role].append(v)
        return retq
    def _de2te(entities):
        retq={}
        j=0
        for k,vs in entities.items():
            for v in vs:
                retq[j] = v
                j += 1
        return retq
    roleMap={}
    retq=[]
    for e in parses:
        intent=e["intent"]
        entities=_te2de(e["entities"])
        for role, fillers in entities.items():
            if needsGuessFn(set([x["value"] for x in fillers])):
                for guessedRole in {role}.union(context["role2Roles"].get(role,[])):
                    if guessedRole in roleMap:
                        entities[role]=roleMap[guessedRole]
                        break
            elif 0<len(fillers):
                roleMap[role]=fillers
        retq.append({"sentence":e["sentence"],"intent":intent,"entities":_de2te(entities)})
    return retq

def semanticLabelling(text, context):
    text = text.strip()
    text = rainDance(text)
    intentUtterances=splitIntents(text, context)
    parsedIntents=degroup([parseIntent(x, context) for x in intentUtterances])
    parsedIntents=guessRoles(parsedIntents, context, lambda x: 0!=len(x.intersection(placeholderWords)))
    for k,e in enumerate(parsedIntents):
        if (0==len(e["entities"])) and (k < len(parsedIntents)-1):
            j=0
            for espec in parsedIntents[k+1]["entities"].values():
                 role = espec.get("role", "UndefinedRole")
                 if role in context["intent2Roles"].get(e["intent"], {}):
                     e["entities"][j] = espec.copy()
                     e["entities"][j]["idx"] = j
                     j+=1
    return parsedIntents

