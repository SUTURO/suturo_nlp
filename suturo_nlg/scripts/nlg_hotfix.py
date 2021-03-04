#!/usr/bin/env python

# generall ROS stuff
import rospy
# interface with knowledge
import rosprolog_client
# Messages
import actionlib
import nlg_msgs.msg

from tmc_msgs.msg import TalkRequestActionGoal
from tmc_msgs.msg import Voice
from std_msgs.msg import String

def callback(data):
    msg = TalkRequestActionGoal()
    voice = Voice()
    voice.sentence = data.feedback.feedback.data
    msg.goal.data = voice
    pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('nlg_hotfix', anonymous=True)
    pub = rospy.Publisher('/talk_request_action/goal', TalkRequestActionGoal, queue_size=10)
    rospy.Subscriber("/nlg_requests/feedback", nlg_msgs.msg.LanguageGenerationActionFeedback, callback)
    rospy.spin()

