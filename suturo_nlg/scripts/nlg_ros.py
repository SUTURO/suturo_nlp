#!/usr/bin/env python
# generall ROS stuff and MSGS
import rospy
import rospkg
from std_msgs.msg import String
from nlg_msgs.msg import MeaningRepresentation, KeyValuePair
import tmc_msgs.msg
from tmc_msgs.msg import Voice

# used for starting the nlg.py3 script in diffrent thread
import os
import threading

# connection to nlg.py3 script
import zmq
from tinyrpc import RPCClient
from tinyrpc.protocols.jsonrpc import JSONRPCProtocol
from tinyrpc.transports.zmq import ZmqClientTransport
import json
ctx = zmq.Context()

rpc_client = RPCClient(
    JSONRPCProtocol(),
    ZmqClientTransport.create(ctx, 'tcp://127.0.0.1:5002')
)


gen_server = rpc_client.get_proxy()


def callback(data):
    # create String
    list_send = []
    for kvp in data.role_values:
                
        list_send.append((kvp.key, kvp.value))
        # TODO get Object/Surface spoken name

    talk_sentence = gen_server.generate_text(list_send) # call the remote function
    talk_request_pub.publish(Voice(sentence=talk_sentence, language=tmc_msgs.msg.Voice.kEnglish))
    # taken from /hsrb_battery_notifier/battery_notifier.py I don't know what the k stands for either


def start_py3_script():
    rospack = rospkg.RosPack()
    wd = rospack.get_path('suturo_nlg') + "/scripts"
    # subprocess.Popen('python3 nlg.py3', cwd=wd, shell=True)
    os.system("python3 " + wd + "/nlg.py3") # + "&> /dev/null"



if __name__ == '__main__':
    talk_request_pub = rospy.Publisher('/talk_request', Voice, queue_size=1)
    rospy.Subscriber("nlg_requests", MeaningRepresentation, callback)
    rospy.init_node('natural_language_generator', anonymous=True)
    
    # starting nlg.py3
    x = threading.Thread(target=start_py3_script, args=())
    x.daemon = True
    x.start()

    '''
    mr = MeaningRepresentation
    mr.role_values
    kvp = KeyValuePair()
    kvp.key = "action"
    kvp.value = "place"
    mr.role_values = [kvp]
    kvp1 = KeyValuePair()
    kvp1.key = "object_id"
    kvp1.value = "the red pringles can"

    kvp2 = KeyValuePair()
    kvp2.key = "goal_surface_id"
    kvp2.value = "bookshelf"

    kvp3 = KeyValuePair()
    kvp3.key = "goal_room_id"
    kvp3.value = "living room"
    mr.role_values = [kvp, kvp1, kvp2, kvp3]
    callback(mr)
    '''

    # run forever and wait for incoming msgs
    rospy.spin()

