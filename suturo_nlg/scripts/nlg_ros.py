#!/usr/bin/env python

# generall ROS stuff
import rospy
# interface with knowledge
import rosprolog_client
# Messages
import actionlib
import nlg_msgs.msg
# connection to nlg.py3 script
import zmq
from tinyrpc import RPCClient, RPCProxy
from tinyrpc.protocols.jsonrpc import JSONRPCProtocol
from tinyrpc.transports.zmq import ZmqClientTransport
from std_msgs.msg import String

# Start an RPC Client
rpc_client = RPCClient(
    JSONRPCProtocol(),
    ZmqClientTransport.create(zmq.Context(), 'tcp://127.0.0.1:5002')
)
gen_server = None

prolog = rosprolog_client.Prolog()


class NlgAction:
    # create messages that are used to publish feedback/result
    _feedback = nlg_msgs.msg.LanguageGenerationFeedback()
    _result = nlg_msgs.msg.LanguageGenerationResult()

    def __init__(self, name):
        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, nlg_msgs.msg.LanguageGenerationAction,
                                                execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

    def execute_cb(self, goal):
        list_send = []  # can be converted to JSON for the RPC call
        for kvp in goal.key_value_pairs:
            kvp.value = knowledge_translate(kvp.key, kvp.value)
            list_send.append((kvp.key, kvp.value))

        value_from_s_nlg = gen_server.generate_text(list_send)
        if value_from_s_nlg.startswith("!ERROR!~"):
            error_msg = value_from_s_nlg.split("~")[1]
            rospy.logerror(error_msg)
            sentence = "An error occurred in the natural language generation pipeline. The error given was: " + error_msg

        else:
            sentence = value_from_s_nlg.encode('ascii', 'ignore')

        rosstring = String()
        rosstring.data = sentence
        self._feedback.feedback = rosstring
        self._as.publish_feedback(self._feedback)
        self._result.generated_sentence = rosstring
        self._as.set_succeeded(self._result)


#  @staticmethod
def knowledge_translate(id_type, thing_id):
    prolog_query = ""
    if id_type == "object_id" or id_type == "object_id_2":
        prolog_query = "object_tts('http://www.semanticweb.org/suturo/ontologies/2020/3/objects#" + thing_id + "', Name)"    
    elif id_type == "start_surface_id" or id_type == "goal_surface_id":
        #  TODO this has to be checked once knowledge is done
        prolog_query = "surface_tts(" + thing_id + ", Name)" 
    elif id_type == "start_room_id" or id_type == "goal_room_id":
        #  TODO this has to be checked once knowledge is done
        prolog_query = "room_tts(" + thing_id + "', Name)"
    if not prolog_query == "":
        solutions = prolog.all_solutions(prolog_query)
        return solutions[0]['Name']
    return thing_id


if __name__ == '__main__':
    rospy.init_node('natural_language_generator', anonymous=True)
    rospy.loginfo("NLG_ROS: Connecting to nlg.py3")
    gen_server = rpc_client.get_proxy()
    rospy.loginfo(gen_server.test_connection())
    # run forever and wait for incoming msgs
    server = NlgAction("nlg_requests")
    rospy.spin()

