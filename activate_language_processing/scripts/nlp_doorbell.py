#!/usr/bin/env python3

from argparse import ArgumentParser

import numpy as np
import rclpy
import scipy as sp
import scipy.io.wavfile
from audio_common_msgs.msg import AudioStamped  # Not more in use?
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String, UInt8MultiArray

AudioMsg = AudioStamped


class DoorbellDetectionNode(Node):
    """
    A node that detects a doorbell ring to start the challenges.
    When a doorbell is detected, we publish a string via topic.
    """

    def __init__(self, args):
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # Publisher
        super().__init__("doorbell_detection")
        self.publisher = self.create_publisher(String, args.outputTopic, 1)

        # Subscriber for microphone data (/audio/audio is HSR?)
        self.subscription = self.create_subscription(
            # topic: "/audio" not yet finalized.
            AudioMsg,
            "/audio",
            self.callback_fft,
            qos,
        )
        self.get_logger().info("Doorbell detection started")

        # Read the reference audio
        _, self.reference = scipy.io.wavfile.read(args.reference)
        # Create a new buffer with the length of the reference data
        self.buffer = np.zeros(len(self.reference), dtype=np.int16)
        self.divider = 0

        self.threshold = float(args.threshold)

    def callback_fft(self, data: AudioMsg) -> None:
        """
        Every time new data arrives form our microphone this function will be executed.
        It reads the new microphone data and concatenates it to the buffer.
        Every 100th concatenation we compare the data with the reference.

        Arguments:
            data: New audio data from the microphone.
        """
        # audio.audio_data for AudioStamped
        audio_data = data.audio.audio_data
        new_data = np.array(audio_data.int16_data, dtype=np.int16)
        # remove the len(new_data) old data and append the new data -> keep the same buffer size
        self.buffer = np.concatenate([self.buffer[len(new_data) :], new_data])
        # FIXME: for debug; maybe remove later
        self.get_logger().info(f"New data received: {self.buffer}")

        self.divider += 1
        if self.divider >= 100:
            self.divider = 0
            self.compare()

    def compare(self) -> None:
        """
        Compare data with reference audio using fft.
        """
        comp = sp.signal.fftconvolve(self.buffer, self.reference, mode="valid")
        calc = np.multiply(comp.max(), np.float64(1e-9))

        if calc > self.threshold:
            msg = String()
            msg.data = "Doorbell detected!"
            self.publisher.publish(msg)
            self.get_logger().info("Doorbell detected!")

            # TODO: Maybe stop Node if detected?


def main() -> None:
    rclpy.init()

    parser = ArgumentParser()
    parser.add_argument(
        "-r",
        "--reference",
        default="db2_16K.wav",
        help="File path (recommended: absolute, not relative path) to a WAV file storing the reference doorbell. "
        "This WAV MUST be mono (not stereo!), 16ksps, 16bit/sample.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        default=10,
        help="Threshold to declare the input sound similar enough to the reference doorbell. Default: 10.0",
    )
    parser.add_argument(
        "-o",
        "--outputTopic",
        default="bell_rang",
        help="Name of topic to send notifications to. Default: bell_rang",
    )
    parsed_args = parser.parse_args()

    node = DoorbellDetectionNode(parsed_args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
