import json
import subprocess
import sys

import requests
import spacy


def install_spacy_required_packages():
    packages = ["en_core_web_sm"]
    for package_name in packages:
        if not spacy.util.is_package(package_name):
            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", package_name]
            )


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
    return (idxS <= idx) and (idx < idxE)


def rainDance(text):
    """
    Perform rain dances to hopefully appease the Neural Network gods
    so as to get a good parse out of a text.

    In boring speak, a few preprocessing steps that MIGHT steer spacy
    away from some dumb failures.
    """
    # Avoid a then-clause -advcls-> previous clause, should instead have previous clause -dep|conj-> then-clause
    text = text.replace(", then", " then")
    text = text.replace("'d", " would")
    text = text.replace(",", " ,")
    text = text.replace("'ll", " will")

    return text


def getAttributes(idxS, idxE, idx2Tok, deps, poss):

    attributes = []

    for idx, tok in idx2Tok.items():
        if not inRange(idx, idxS, idxE):
            continue

        for c in tok.children:
            if (
                (not inRange(c.idx, idxS, idxE))
                and (c.dep_ in deps)
                and (c.pos_ in poss)
            ):
                _, text, _ = getSubtree(c)
                attributes.append(text)

    return tuple(attributes)


def subtreeDep(idxS, idxE, idx2Tok):
    inSpan = set()
    idx2Dep = {}

    for idx, tok in idx2Tok.items():
        if inRange(idx, idxS, idxE):
            inSpan.add(idx)
            idx2Dep[idx] = (tok.head.idx, tok.dep_)

    for idx, (hIdx, dep) in idx2Dep.items():
        if hIdx not in inSpan:
            return dep

    return None


def parseIntent(cspec, context):
    """
    Use RASA to parse a simple sentence (one intent).
    """
    text = cspec["text"]
    sStart = cspec["start"]
    idx2Tok = cspec["idx2Tok"]

    req = {"text": text}
    r = requests.post(context["rasaURI"], data=bytes(json.dumps(req), "utf-8"))
    response = json.loads(r.text)

    result = {"sentence": text, "intent": response["intent"]["name"], "entities": {}}

    for k, e in enumerate(response["entities"]):
        # print("Entity", e, sStart)
        eStart = e.get("start", 0) + sStart
        eEnd = e.get("end", 0) + sStart

        if subtreeDep(eStart, eEnd, idx2Tok) in roleForbiddenDeps:
            continue

        result["entities"][k] = {
            "idx": k,
            "role": e.get("role", "UndefinedRole"),
            "value": e.get("value", "UnparsedEntity"),
            "group": int(e.get("group", 0)),
            "entity": e.get("entity", "owl:Thing"),
            "propertyAttribute": getAttributes(
                eStart, eEnd, idx2Tok, attrDeps, propAttrPOS
            ),
            "actionAttribute": getAttributes(
                eStart, eEnd, idx2Tok, attrDeps, actAttrPOS
            ),
            "numberAttribute": getAttributes(
                eStart, eEnd, idx2Tok, numDeps, numAttrPOS
            ),
        }

    return result


def degroup(parses):
    """
    Convert a parse that may have groups (sets of entities to act on in the same way in parallel) into a list
    of parses.
    """
    print(parses)
    results = []

    for e in parses:
        intent = e["intent"]
        entities = e["entities"]

        groups = {0: {}}

        for k, ed in entities.items():
            role = ed["role"]
            group = ed["group"]

            if group not in groups:
                groups[group] = {}
            if role not in groups[group]:
                groups[group][role] = []

            groups[group][role].append(ed)

        for k in sorted(groups.keys()):
            eds = {}

            for r, vs in groups[k].items():
                for v in vs:
                    v_copy = v.copy()
                    v_copy["group"] = 0
                    eds[v["idx"]] = v_copy

                    # eds[v["idx"]] = v.copy()
                    # eds[v["idx"]]["group"] = 0

            results.append(
                {"sentence": e["sentence"], "intent": intent, "entities": eds}
            )

    return results


def getSubtree(tok):
    """
    Return the subtree of a token, but stop at dependent verbs.
    This allows splitting a text into sentences.
    """
    inText = [(tok.idx, tok)]
    todo = list(tok.children)
    next = []
    idx2Tok = {tok.idx: tok}
    excluded = set()

    for c in tok.children:
        # if (("VERB" == c.pos_) and (c.dep_ in conjDeps)) or (("AUX" == c.pos_ ) and ((c.dep_ in conjDeps) or (c.dep_ in auxDeps)) and ("be" == c.lemma_)):
        if ("VERB" == c.pos_) and (c.dep_ in conjDeps):
            next.append(c)
            excluded.add(c.idx)

    while todo:
        cr = todo.pop()

        if cr.idx not in excluded:
            inText.append((cr.idx, cr))
            idx2Tok[cr.idx] = cr
            todo.extend(list(cr.children))

    toks = sorted(inText, key=lambda x: x[0])
    subtree_t = " ".join([str(token) for idx, token in toks])

    return next, subtree_t, idx2Tok


def splitIntents(text, context):
    doc = context["nlp"](text)
    intentUtterances = []

    for s in doc.sents:
        todo = [s.root]

        while todo:
            cr = todo.pop()
            next, text, idx2Tok = getSubtree(cr)
            todo.extend(next)

            intentUtterances.append(
                {"text": text, "start": min(idx2Tok.keys()), "idx2Tok": idx2Tok}
            )

    return intentUtterances


def guessRoles(parses, context, needsGuessFn):

    def _te2de(entities):
        results = {}
        for k, v in entities.items():
            role = v["role"]
            if role not in results:
                results[role] = []
            results[role].append(v)

        return results

    def _de2te(entities):
        results = {}
        j = 0

        for k, vs in entities.items():
            for v in vs:
                results[j] = v
                j += 1

        return results

    roleMap = {}
    results = []

    for e in parses:
        intent = e["intent"]
        entities = _te2de(e["entities"])

        for role, fillers in entities.items():
            if needsGuessFn(set([x["value"] for x in fillers])):
                for guessedRole in {role}.union(context["role2Roles"].get(role, [])):
                    if guessedRole in roleMap:
                        entities[role] = roleMap[guessedRole]
                        break
            elif len(fillers) > 0:
                roleMap[role] = fillers

        results.append(
            {"sentence": e["sentence"], "intent": intent, "entities": _de2te(entities)}
        )

    return results


def semanticLabelling(text, context):
    text = text.strip()
    text = rainDance(text)

    intentUtterances = splitIntents(text, context)
    parsedIntents = degroup([parseIntent(x, context) for x in intentUtterances])
    parsedIntents = guessRoles(
        parsedIntents, context, lambda x: 0 != len(x.intersection(placeholderWords))
    )

    for k, e in enumerate(parsedIntents):
        no_entities = len(e["entities"]) == 0
        not_last = k < len(parsedIntents) - 1

        if no_entities and not_last:
            next_ent = parsedIntents[k + 1]["entities"]
            intent_roles = context["intent2roles"].get(e["intent"], {})

            j = 0
            for espec in next_ent.values():
                role = espec.get("role", "UndefinedRole")

                if role in intent_roles:
                    ent_copy = espec.copy()
                    ent_copy["entities"][j] = ent_copy
                    ent_copy["idx"] = j
                    j += 1
                    # e["entities"][j] = espec.copy()
                    # e["entities"][j]["idx"] = j
                    # j += 1

    return parsedIntents
