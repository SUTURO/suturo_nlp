#! /home/hawkin/envs/whisper/bin/python3.8

# Authors: Vanessa Hassouna https://github.com/sunava/

import rospy
from sound_play.msg import SoundRequestActionGoal, SoundRequest


class SoundRequestPublisher:
    """
    A class to publish sound requests in a ROS environment.
    """

    current_subscriber: rospy.Subscriber = None
    """
    Reference to the current subscriber instance.
    """

    def __init__(self, topic='/sound_play/goal', queue_size=10, latch=True):
        """
        Initializes the SoundRequestPublisher with a ROS publisher.

        :param topic: The ROS topic to publish sound requests to. Default is '/sound_play/goal'.
        :param queue_size: The size of the message queue for the publisher. Default is 10.
        :param latch: Whether the publisher should latch messages. Default is True.
        """
        self.pub = rospy.Publisher(topic, SoundRequestActionGoal, queue_size=queue_size, latch=latch)
        self.msg = SoundRequestActionGoal()
        self.msg.goal.sound_request.sound = 1
        self.msg.goal.sound_request.command = 1
        self.msg.goal.sound_request.volume = 2.0

    def publish_sound_request(self):
        """
        Publish the sound request message.
        """
        rospy.loginfo("[BEEP] Publishing sound request")
        rospy.loginfo("[BEEP] Waiting for subscribers to connect...")
        while self.pub.get_num_connections() == 0:
            rospy.sleep(0.1)  # Sleep for 100ms and check again
        self.pub.publish(self.msg)
        rospy.loginfo("[BEEP] Sound request published")


