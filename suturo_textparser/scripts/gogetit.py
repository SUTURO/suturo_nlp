#!/usr/bin/env python

import re
from parsetron import Set, Regex, Optional, OneOrMore, Grammar, RobustParser
import rospy
from nlp_msgs.msg import GoAndGetIt
from std_msgs.msg import String
import rosprolog_client


prolog = rosprolog_client.Prolog()

def get_items():
    rospy.init_node('suturo_GoAndGetIt_Parser', anonymous=True)
    query = "all_obj_names(N)."
    solutions = prolog.all_solutions(query)
    names = []
    for name in solutions[0]['N']:
        names.append(name.__str__()[1:-1])
    return Set(names)


class TransportGrammar(Grammar):
    article = Set(['a', 'an', 'the', 'any', 'some'])
    personLeft = Set(['person left'])
    personRight = Set(['person right'])
    beneficiary = personLeft | personRight
    item = get_items()
    to = Set(['to'])
    color = Set(['black','blue','brown','cyan','green','magenta','orange','pink','purple','red','teal','yellow','white'])
    itemCol = Optional(color) + item
    itemArt = Optional(article) + itemCol #  | it
    destination = to + beneficiary
    command = itemArt + Optional(destination)
    GOAL = OneOrMore(command)


def search_benificiary(node):
    string = node.__str__()
    if "personLeft" in string:
        return "left"
    elif "personRight" in string:
        return "right"
    else:
        return "both"
    


def recursive_search_object(node):
    if node.is_leaf() and node.__str__().startswith("(item"):
        regresult = re.findall(r"\"([A-Za-z0-9_]+)\"\)", node.__str__())
        return " ".join(regresult)
    elif node.is_leaf():
        return ""
    else:
        a = []
        for c in node.children:          
            a.append(recursive_search_object(c))
        return "".join(a)


def callback(data):
    parser = RobustParser(TransportGrammar())
    tree, result = parser.parse(data.data)
    msg = GoAndGetIt()
    if tree:
        benificiary = search_benificiary(tree)
        if benificiary == "left":
            msg.person_left = True
        elif benificiary == "right":
            msg.person_right = True
        else:
            msg.person_left = True
            msg.person_right = True
        msg.perceived_object_name = recursive_search_object(tree)
    
    pub.publish(msg)


if __name__ == '__main__':
    try:
        # Start the publisher node
        # rospy.init_node('suturo_GoAndGetIt_Parser', anonymous=True)
        pub = rospy.Publisher('sp_output', GoAndGetIt, queue_size=10)
        # Start the subscriber node
        rospy.Subscriber('sp_input', String, callback)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
