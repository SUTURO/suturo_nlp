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

placeholder_words = {"her", "him", "it", "them", "there"}
conj_deps = {"conj", "dep"}
aux_deps = {"ccomp"}
attr_deps = {"acl", "amod", "relcl"}
num_deps = {"nummod"}
act_attr_pos = {"VERB"}
prop_attr_pos = {"ADJ", "ADV"}
num_attr_pos = {"NUM"}
role_forbidden_deps = attr_deps.union(num_deps)


def in_range(idx, start_index, end_index):
    return (start_index <= idx) and (idx < end_index)


def rain_dance(text):
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


def get_attributes(start_index, end_index, index_to_token, dependency, poss):
    attributes = []
    for idx, token in index_to_token.items():
        if not in_range(idx, start_index, end_index):
            continue

        for child in token.children:
            if in_range(child.idx, start_index, end_index):
                continue
            if child.dep_ not in dependency:
                continue
            if child.pos_ not in poss:
                continue

            _, text, _ = get_subtree(child)
            attributes.append(text)

    return tuple(attributes)


def subtree_dep(start_index, end_index, index_to_token):
    in_span = set()
    idx_to_dep = {}

    for idx, token in index_to_token.items():
        if in_range(idx, start_index, end_index):
            in_span.add(idx)
            idx_to_dep[idx] = (token.head.idx, token.dep_)

    for idx, (hIdx, dep) in idx_to_dep.items():
        if hIdx not in in_span:
            return dep

    return None


def parse_intent(cspec, context):
    """
    Use RASA to parse a simple sentence (one intent).
    """
    text = cspec["text"]
    sentence_start = cspec["start"]
    index_to_token = cspec["index_to_token"]
    req = {"text": text}
    r = requests.post(context["rasaURI"], data=bytes(json.dumps(req), "utf-8"))
    response = json.loads(r.text)

    result = {"sentence": text, "intent": response["intent"]["name"], "entities": {}}

    for k, entity in enumerate(response["entities"]):
        # print("Entity", entity, sentence_start)
        entity_start = entity.get("start", 0) + sentence_start
        entity_end = entity.get("end", 0) + sentence_start

        if subtree_dep(entity_start, entity_end, index_to_token) in role_forbidden_deps:
            continue

        result["entities"][k] = {
            "idx": k,
            "role": entity.get("role", "UndefinedRole"),
            "value": entity.get("value", "UnparsedEntity"),
            "group": int(entity.get("group", 0)),
            "entity": entity.get("entity", "owl:Thing"),
            "propertyAttribute": get_attributes(
                entity_start, entity_end, index_to_token, attr_deps, prop_attr_pos
            ),
            "actionAttribute": get_attributes(
                entity_start, entity_end, index_to_token, attr_deps, act_attr_pos
            ),
            "numberAttribute": get_attributes(
                entity_start, entity_end, index_to_token, num_deps, num_attr_pos
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


def get_subtree(token):
    """
    Return the subtree of a token, but stop at dependent verbs.
    This allows splitting a text into sentences.
    """
    in_text = [(token.idx, token)]
    todo = list(token.children)
    next_tokens = []
    index_to_token = {token.idx: token}
    excluded = set()

    for child in token.children:
        # if (("VERB" == child.pos_) and (child.dep_ in conj_deps)) or (("AUX" == child.pos_ ) and ((child.dep_ in conj_deps) or (child.dep_ in aux_deps)) and ("be" == child.lemma_)):
        if ("VERB" == child.pos_) and (child.dep_ in conj_deps):
            next_tokens.append(child)
            excluded.add(child.idx)

    while todo:
        current_token = todo.pop()

        if current_token.idx not in excluded:
            in_text.append((current_token.idx, current_token))
            index_to_token[current_token.idx] = current_token
            todo.extend(list(current_token.children))

    toks = sorted(in_text, key=lambda x: x[0])
    subtree_t = " ".join([str(token) for idx, token in toks])

    return next_tokens, subtree_t, index_to_token


def split_intents(text, context):
    doc = context["nlp"](text)
    intent_utterances = []

    for s in doc.sents:
        todo = [s.root]
        while todo:
            current_token = todo.pop()
            next_tokens, text, index_to_token = get_subtree(current_token)
            todo.extend(next_tokens)
            intent_utterances.append(
                {
                    "text": text,
                    "start": min(index_to_token.keys()),
                    "index_to_token": index_to_token,
                }
            )

    return intent_utterances


def guess_roles(parses, context, needs_guess_fn):
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

    role_map = {}
    results = []

    for parse in parses:
        intent = parse["intent"]
        entities = _te2de(parse["entities"])

        for role, fillers in entities.items():
            if needs_guess_fn(set([x["value"] for x in fillers])):
                for guessedRole in {role}.union(context["role2Roles"].get(role, [])):
                    if guessedRole in role_map:
                        entities[role] = role_map[guessedRole]
                        break
            elif len(fillers) > 0:
                role_map[role] = fillers

        results.append(
            {
                "sentence": parse["sentence"],
                "intent": intent,
                "entities": _de2te(entities),
            }
        )

    return results


def semantic_labelling(text, context):
    text = text.strip()
    text = rain_dance(text)
    intent_utterances = split_intents(text, context)
    parsed_intents = degroup([parse_intent(x, context) for x in intent_utterances])
    parsed_intents = guess_roles(
        parsed_intents, context, lambda x: 0 != len(x.intersection(placeholder_words))
    )

    for k, e in enumerate(parsed_intents):
        no_entities = len(e["entities"]) == 0
        not_last = k < len(parsed_intents) - 1

        if no_entities and not_last:
            next_entities = parsed_intents[k + 1]["entities"]
            intent_roles = context["intent2roles"].get(e["intent"], {})
            j = 0

            for entity_spec in next_entities.values():
                role = entity_spec.get("role", "UndefinedRole")
                if role in intent_roles:
                    e["entities"][j] = entity_spec.copy()
                    e["entities"][j]["idx"] = j
                    j += 1

    return parsed_intents
