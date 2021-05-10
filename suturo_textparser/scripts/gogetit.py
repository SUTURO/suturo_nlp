# !/usr/bin/env python
import re
from parsetron import Set, Regex, Optional, OneOrMore, Grammar, RobustParser
import rospy
from suturo_nlp.msg import *
from std_msgs.msg import String


class TransportGrammar(Grammar):
    article = Set(['a', 'an', 'the', 'any', 'some'])
    actionTriadic = Set(['bring', 'fetch', 'get', 'give', 'pass'])
    beneficiary = Set(['me'])
    room = Set(['outside', 'bar', 'hall', 'attic', 'basement', 'bathroom', 'bedroom', 'cellar', 'dining area', 'dining room', 'garden', 'greenhouse', 'guest bathroom', 'kitchen', 'larder', 'living room', 'lobby', 'pantry', 'winter garden'])
    item = Set(['cup', 'fishcan', 'mug', 'pringles can'])
    politeStart = Set(['could you', 'could you please', 'please', 'pretty please', 'would you', 'would you please'])
    politeEnd = Set(['please', 'pretty please'])
    frm = Set(['from'])
    to = Set(['to'])
    it = Set(['it'])
    in_ = Set(['in'])
    there_is = Set(['There is'])

    color = Set(['black','blue','brown','cyan','green','magenta','orange','pink','purple','red','teal','yellow','white'])
    itemCol = Optional(color) + item
    roomArt = Optional(article) + room
    itemArt = Optional(article) + itemCol #  | it
    source = frm + roomArt
    destination = to + beneficiary
    transport2Beneficiary1 = actionTriadic + beneficiary + itemArt + Optional(source)
    transport2Beneficiary2 = there_is + itemArt + in_ + roomArt + actionTriadic + it + destination
    command = transport2Beneficiary1 | transport2Beneficiary2
    commandPol = Optional(politeStart) + command + Optional(politeEnd)
    GOAL = OneOrMore(commandPol)


def recursive_search_room(node):
    if node.is_leaf() and node.__str__().startswith("(room"):
        regresult = re.findall(r"\"([A-Za-z0-9_]+)\"\)", node.__str__())
        return " ".join(regresult)
    else:
        for c in node.children:
            recursive_search_room(c)


def recursive_search_object(node):
    if not node.is_leaf():
        if node.__str__().startswith("(itemCol"):
            regresult = re.findall(r"\"([A-Za-z0-9_]+)\"\)", node.__str__())
            return " ".join(regresult)
        else:
            for c in node.children:
                recursive_search_object(c)


def callback(data):
    parser = RobustParser((TransportGrammar()))
    tree, result = parser.parse(data.data)
    msg = suturo_nlp.msg.FetchRequest()
    msg.perceived_room_name = recursive_search_room(tree)
    msg.perceived_object_name = recursive_search_object(tree)
    pub.publish(msg)

if __name__ == '__main__':
    try:
        # Start the publisher node
        rospy.init_node('suturo_GoAndGetIt_Parser', anonymous=True)
        pub = rospy.Publisher('sp_output', suturo_nlp.msg.FetchRequest, queue_size=10)
        # Start the subscriber node
        rospy.Subscriber('suturo_go_and_get_it_sentence', String, callback)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
