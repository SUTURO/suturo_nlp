#!/usr/bin/env python
import rospy
import rospkg
from tmc_rosjulius_msgs.msg import RecognitionResult
from nlp_msgs.msg import StaticCommand
from std_msgs.msg import String


def callback(data):
    sentence = find_first(data)
    if not sentence == "" and not is_static(sentence):
        sc_pub.publish(sentence)


def is_static(sentence):
    if is_stop(sentence):
        command = StaticCommand()
        command.command = StaticCommand.STOP
        hc_pub.publish(command)
        return True
    elif is_start(sentence):
        command = StaticCommand()
        command.command = StaticCommand.START
        hc_pub.publish(command)
        return True
    elif is_continue(sentence):
        command = StaticCommand()
        command.command = StaticCommand.CONTINUE
        hc_pub.publish(command)
        return True

    return False


def is_stop(sentence):
    return is_in_file(sentence, wd + '/commands/stop.txt')


def is_start(sentence):
    return is_in_file(sentence, wd + '/commands/start.txt')


def is_continue(sentence):
    return is_in_file(sentence, wd + '/commands/continue.txt')


def is_in_file(sentence, filepath):
    with open(filepath) as f:
        if sentence in f.read():
            return True
        else:
            return False


def find_first(data):
    mostliklySentence = data.sentences[0]
    score1 = data.scores[0]

    if len(data.scores) == 1 and score1 > 0.7:
        return mostliklySentence
    elif len(data.scores) > 1:
        score2 = data.scores[1]
        if score1 > 0.85 and score1 - score2 > 0.1:
            return mostliklySentence
    return ""


if __name__ == '__main__':
    try:
        rospack = rospkg.RosPack()
        wd = rospack.get_path('suturo_textparser')
        rospy.init_node('textparser', anonymous=False)
        hc_pub = rospy.Publisher('hard_commands', StaticCommand, queue_size=10)
        sc_pub = rospy.Publisher('to_slingparser', String, queue_size=10)
        rospy.Subscriber('input', RecognitionResult, callback)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
