#!/usr/bin/env python
import rospy
import rospkg
import rosprolog_client
from suturo_nlg.msg import KeyValuePair, MeaningRepresentation

prolog = rosprolog_client.Prolog()
rospack = rospkg.RosPack()
wd = rospack.get_path('suturo_nlg')
filepath = wd + "/scripts/object_translate"


def item_translator_to_knowledge(item):
    with open(filepath) as f:
        for line in f:
            if item == line.split(",")[0]:
                return line.split(",")[1].rstrip("\n")
    pub(errordict("I have no idea what you mean with an object called " + item))
    return False


def item_translator_from_knowledge(item):
    with open(filepath) as f:
        for line in f:
            if str(item) == line.split(",")[1].rstrip("\n"):
                return str(line.split(",")[0])
    return False


def errordict(text="Something went wrong"):
    return {"error": text}


def is_there(dict):
    if not dict.keys().__contains__("item"):
        rospy.logerr("Message didn't contain item")
        datadict = errordict("I didn't get the object you meant.")
        pub(datadict)
        return
    if not dict.keys().__contains__("location"):
        rospy.logerr("Message didn't contain location")
        datadict = errordict()
        pub(datadict)
        return
    item_t = item_translator_to_knowledge(dict["item"])
    if not item_t:
        pub(errordict("Item not known in the knowledge base"))
        return
    item_t = item_translator_to_knowledge(dict["item"])
    query_string = "rdfs_individual_of(_X,hsr_objects:'" + item_t + "'), rdf_has(_X, hsr_objects:'supportedBy',_Link), rdf_urdf_name(_Link,Name)."
    rospy.loginfo("Query for supposed to go:\n" + query_string)
    solutions = prolog.all_solutions(query_string)
    rospy.loginfo(solutions)

    if not solutions:
        dict["answer"] = "no"

    location = dict["location"].replace(" ", "_")

    # TODO change this to a diffrent file and expand
    if location == "big_table":
        location = "table_center"
    elif location == "hcr_shelf":
        location = "hcr_shefl"
    print(solutions)
    for solution in solutions:
        if solution['Name'].__contains__(location):
            dict["answer"] = "yes"
            pub(dict)
            return
    dict["answer"] = "no"
    pub(dict)


def what_is_on(dict):
    if not dict.keys().__contains__("location"):
        rospy.logerr("Message didn't contain location")
        pub(errordict("What location did you mean?"))
        return False

    if dict["location"] == "hcr shelf":
        query_string = "hsr_existing_objects(_Objs), member(Obj,_Objs), rdf_has(Obj, hsr_objects:'supportedBy', _S), rdf_urdf_name(_S,_SN), sub_string(_SN,_,_,_,\"hcr_shefl\")."
    elif dict["location"] == "bookshelf":
        query_string = "hsr_existing_objects(_Objs), member(Obj,_Objs), rdf_has(Obj, hsr_objects:'supportedBy', _S), rdf_urdf_name(_S,_SN), sub_string(_SN,_,_,_,\"bookshelf_floor\")."
    elif dict["location"] == "big table":
        query_string = "once((rdf_urdf_name(_Table, table_center),rdf_has(Obj, hsr_objects:'supportedBy', _Table)))."
    elif dict["location"] == "small table":
        query_string = "once((rdf_urdf_name(_Table, bed_table_center),rdf_has(Obj, hsr_objects:'supportedBy', _Table)))."
    elif dict["location"] == "ground":
        query_string = "once((rdf_has(hsr_objects:'supportedBy', ground)))."
    else:
        rospy.logerr("Message didn't contain location")
        dict = errordict("What location did you mean?")
        pub(dict)
        return False
    solutions = prolog.all_solutions(query_string)
    if not solutions:
        pub(errordict("I don't know of any objects on the " + dict["location"]))
        return
    print(solutions[0]['Obj'].split('#')[1].split('_')[0])
    # dict["item"] = str(item_translator_from_knowledge(solutions[0]['Obj'].split('#')[1].split('_')[0]))
    dict["item"] = solutions[0]['Obj'].split('#')[1].split('_')[0]
    pub(dict)
    return True


def supposed_to_go(dict):
    if not dict.keys().__contains__("item"):
        rospy.logerr("Message didn't contain item")
        datadict = errordict("What item did you mean?")
        pub(datadict)
        return
    item_t = item_translator_to_knowledge(dict["item"])
    query_string = "hsr_existing_objects(_Objs), member(_Obj,_Objs), rdfs_individual_of(_Obj, hsr_objects:'" + item_t + "'), object_goal_surface(_Obj,_Surf), rdf_urdf_name(_Surf,Name)."
    rospy.loginfo("Query for supposed to go:\n" + query_string)
    solutions = prolog.all_solutions(query_string)
    rospy.loginfo(solutions)
    if not solutions or solutions == []:
        pub(errordict("Didn't get any solution"))
        return
    dict["location"] = solutions[0]['Name'].split('_')[0]
    pub(dict)


def callback(data):
    datadict = {}
    for keyvaluep in data.role_values:
        datadict[keyvaluep.key] = keyvaluep.value

    if datadict.keys().__contains__("predicate"):
        if datadict["predicate"] == "what is on":
            what_is_on(datadict)
        elif datadict["predicate"] == "is there":
            is_there(datadict)
        elif datadict["predicate"] == "supposed to go":
            supposed_to_go(datadict)
        else:
            rospy.logerr("Message didn't contain any known predicates")
            datadict = errordict("The last sentence was not a question that I can understand")
            pub(datadict)
            return
    else:
        rospy.logerr("Message didn't contain any predicates")


def pub(datadict):
    if datadict.keys().__contains__("predicate"):
        datadict["question"] = datadict["predicate"]
        del datadict["predicate"]

    dict_meaning_rep = MeaningRepresentation()
    for key in datadict:
        nkvp = KeyValuePair()
        nkvp.key = key
        nkvp.value = datadict[key]
        dict_meaning_rep.role_values.append(nkvp)
    publisher.publish(dict_meaning_rep)


if __name__ == '__main__':
    rospy.init_node('knowledge_interface')
    rospy.Subscriber("kinterface_in", MeaningRepresentation, callback)
    publisher = rospy.Publisher('kinterface_out', MeaningRepresentation, queue_size=10)
    rospy.loginfo("Knowledge Interface: Waiting for rosprolog service")
    rospy.wait_for_service('/rosprolog/query')
    rospy.loginfo("Knowledge Interface: rosprolog service is running knowledge interface now running")
    rospy.spin()
