import audioop
import collections
import json
import sys
import threading
import warnings
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional

import librosa
import noisereduce as nr
import numpy as np
import rclpy
import soundfile as sf
import spacy
# from audio_common_msgs.msg import AudioData
import speech_recognition as sr
import torch
from activate_language_processing.nlp import semantic_labelling  # type: ignore
from faster_whisper import WhisperModel
# import whisper
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from speech_recognition import AudioData, Recognizer
from std_msgs.msg import String, UInt8MultiArray

from nlp_challenges import *

AudioMsg = UInt8MultiArray  # Define AudioMsg as UInt8MultiArray for ROS2 compatibility

warnings.filterwarnings(
    "ignore", message="FP16 is not supported on CPU; using FP32 instead"
)

# ===== Loading Models =====

device = "cuda" if torch.cuda.is_available() else "cpu"
comp_type = "float16" if device == "cuda" else "int8"

# Using faster-whisper
model = WhisperModel(
    "small",
    device=device,
    compute_type=comp_type,
)

# Load the Whisper model for transcription
# model = whisper.load_model("small.en")

# ==========================


# ----- Constants -----
SAMPLE_RATE = 16000
CHUNK_SIZE = 32000  # 16kHz = 2 Sec
SAMPLE_WIDTH = 2  # 2 bytes
START_SILENCE = 0.2


@dataclass
class Context:
    """
    Configurations about the node, audio and other data.

    Attributes:
        node (Node): ROS2 node instance
        pub (Publisher): ROS2 publisher. Default from args: 'nlp_out'
        stt (Publisher): ROS2 publisher for Speech-to-text. Default from args: 'whisper_out'
        nluURI (str): URI for the Rasa model. Default from args: http://localhost:5005/model/parse
        useHSR (bool): Whether to use the HSR mic
        useAudio (bool): Whether to use an audio file
        audio (str): Audio path or './' for microphone
        lock (threading.Lock): Lock to avoid race conditions
        nlp (spacy.language.Language): Language model
        data (np.ndarray): Audio data
        queue (Queue): Queue of audio data
        transcriber (threading.Thread): Thread for transcription
        listening (bool): Boolean if audio input is currently captured
    """

    node: Node
    pub: Publisher
    stt: Publisher

    nluURI: str
    useHSR: bool = False
    useAudio: bool = False
    audio: str = "./"
    lock: threading.Lock = field(default_factory=threading.Lock)

    nlp: spacy.language.Language = spacy.load("en_core_web_sm")

    data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int16))
    queue: Queue = field(default_factory=Queue)
    transcriber: Optional[Thread] = None
    listening: bool = False

    # speaking: bool = False

    # Currently not is use. nlp.py maybe needs an update
    intent2roles: dict = field(default_factory=dict)
    role2Roles: dict = field(default_factory=dict)


def try_node_logger(msg: str, context: Context):
    """
    Log message with ROS2 node logger. Otherwise, use print

    Args:
        msg: Message to log
        context: Shared context
    """
    if isinstance(context.node, Node):
        context.node.get_logger().info(msg)
    else:
        print(msg)


def is_transcribing(context: Context) -> bool:
    """
    Checks whether a transcription process is currently active and running.
    Returns True if a transcriber object exists and is actively running (i.e., the transcription process is ongoing), and False otherwise.

    Args:
        context: Shared context
    """
    return (context.transcriber is not None) and (context.transcriber.is_alive())


def transcribe_audio(temp_fp: str, prompt: str | None) -> str:
    """
    Transcribing audio with (faster-) whisper.
    """
    segments, _ = model.transcribe(
        temp_fp,
        language="en",
        initial_prompt=prompt,
        beam_size=5,
        # condition_on_previous_text=True,
        # without_timestamps=True,
        vad_filter=True,
    )
    text = "".join(s.text.strip() for s in segments)

    return text


class NLU:
    """
    Natural Language Understanding (NLU).

    Handles semantic parsing of text.
    """

    def __init__(self, context: Context):
        self.context = context

    def nlu_internal(self, text: str, temp_fp: str | Path) -> None:
        """
        Process a text input to extract semantic information like intent and entities,
        formats the extracted data, and publishes it as JSON to a ROS topic.

        Args:
            text: The input text to be analyzed.
            temp_fp: Path to the audio file.
        """
        # Lock so only one thread may execute this code at a time
        with self.context.lock:
            # Check for names and noun and get the fitting prompt
            prompt = self._get_prompts(text)

            try_node_logger(
                f"Using Prompt: '{prompt}'",
                self.context,
            )

            # Transcribe the audio file using Whisper with an initial prompt
            # result = model.transcribe(temp_fp, initial_prompt=prompt)
            # text = result["text"]

            # using faster-whisper:
            text = transcribe_audio(temp_fp, prompt)

            # Analyze text and return parses (a structured object like a dictionary)
            parses = semantic_labelling(
                text,
                # nlp.py wants a dict
                {
                    "nlp": self.context.nlp,
                    "rasaURI": self.context.nluURI,
                    # to avoid errors in nlp.py even if they are empty
                    "role2Roles": self.context.role2Roles,
                    "intent2roles": self.context.intent2roles,
                },
            )

            # Check and publish
            self._process_parses(parses)

    @staticmethod
    def _get_prompts(text: str) -> str:
        """
        Returns the prompt for whisper based on the found names and nouns from `nounDictionary()`.
        """
        names, nouns = nounDictionary(text)

        if not names and not nouns:
            return "The user is ordering food or drinks, or introducing itself."

        if names and nouns:
            return (
                f"The user says their name is one of these: {' , '.join(names)}."
                f"They might like to drink one of these: {' , '.join(nouns)}."
            )

        # Nouns must be empty here
        if names:
            return f"The user says their name is one of these: {' , '.join(names)}."
        else:
            return f"The user wants to order one or several of these food or drink items: {' , '.join(nouns)}."

    def _process_parses(self, parses: list) -> None:
        """
        Processes the parses list to extract semantic information. It checks if the String is emtpy and if special intents are available.
        Afterward publish the result as JSON with the publisher.

        Args:
            parses: The parses from semanticLabelling.
        """
        # Special case intents
        special = {"affirm", "deny", "Callout", "Hobbies", "talk"}

        for p in parses:
            # skipping if sentence or entities list is empty
            if not p["sentence"].strip() or not p["entities"]:
                if p["intent"] not in special:
                    try_node_logger(
                        f"[ALP]: Skipping empty or invalid parse. Sentence: '{p['sentence']}', Intent: '{p['intent']}'",
                        self.context,
                    )
                    continue

            fmt_parse = self._format_parses(p)
            self.context.pub.publish(String(data=json.dumps(fmt_parse)))
            try_node_logger("[ALP]: Done. Waiting for next command.", self.context)

    @staticmethod
    def _format_parses(p: dict) -> dict:
        """
        Formats the parses. We only care about the Sentence, Intent and Entities.

        Args:
            p: The parse to be formatted.

        Returns:
            The formatted parses as a dictionary.
        """
        pAdj = {
            "sentence": p["sentence"],
            "intent": p["intent"],
            "entities": [],  # new empty entities list
        }
        # try_node_logger(f"Entity items: {p['entities'].items()}", self.context)
        print(f"Entity items: {p['entities'].items()}")

        for k, v in p["entities"].items():
            entity_data = v.copy()  # Copy entity’s data dictionary
            entity_data.pop("group")  # Remove metadata that is not needed
            entity_data.pop("idx")  # Remove metadata that is not needed
            pAdj["entities"].append(entity_data)

        return pAdj


class Audio:
    """
    Handling the Audio input.

    To capture audio we can use audio files, a microphone or the HSR microphone.
    Additionally, we handle the silence detection and noise reduction.
    """

    def __init__(self, context: Context):
        self.context = context
        self.nlu = NLU(context)

    def start_listener(self, _msg) -> None:
        """
        Starts a transcription thread if none is currently active.

        Args:
            _msg: The message that triggers the start of the listener.
        """
        try_node_logger("[ALP] got start signal", self.context)

        with self.context.lock:
            if not is_transcribing(self.context):
                # Create a new Thread
                self.context.transcriber = Thread(target=self.transcriber_fn)
                self.context.transcriber.start()

    def transcriber_fn(self) -> None:
        """
        Capture audio from different sources, depending on the giving flag.
        Afterward, use the audio input with the NLU pipeline.
        """
        r = sr.Recognizer()
        r.pause_threshold = 1.0

        if self.context.useHSR:
            audio = self.listen_hsr(r)
        elif self.context.audio == "./":
            audio = self.listen_microphone(r)
        elif self.context.useAudio:
            audio = self.listen_audio()
        else:
            raise ValueError("Invalid audio source configuration.")

        try_node_logger("[WHISPER]: Processing...", self.context)

        if isinstance(audio, sr.AudioData):
            waveform, sample_rate = self._audio_data_to_numpy(audio)
            temp_fp = "/tmp/audio.wav"
            sf.write(temp_fp, waveform, sample_rate)
        else:
            temp_fp = str(audio)

        self.stt_to_nlu(temp_fp)

    def listen_hsr(self, r: Recognizer) -> AudioData:
        try_node_logger("Waiting for the beep...", self.context)
        with self.context.lock:
            self.context.listening = True
        audio = self.listen_to_queue(r)
        with self.context.lock:
            self.context.listening = False
            self.context.data = np.array([], dtype=np.int16)
            self.context.queue = Queue()

        return audio

    def listen_microphone(self, r: Recognizer) -> AudioData:
        with self.context.lock:
            self.context.listening = True
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=1)
            try_node_logger("Speak now...", self.context)
            audio = r.listen(source)
        with self.context.lock:
            self.context.listening = False

        return audio

    def listen_audio(self) -> Path:
        audio_path = self.context.audio
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)
        audio = audio_path

        return audio

    def stt_to_nlu(self, temp_fp: Path | str) -> None:
        # result = model.transcribe(temp_fp, language="en")
        # result = result["text"]

        # using faster-whisper:
        result = transcribe_audio(temp_fp, None)

        try_node_logger("[WHISPER]: Done", self.context)
        print(f"\nWhisper result: {result}")
        self.context.stt.publish(String(data=result))
        self.nlu.nlu_internal(result, temp_fp)

    def listen_to_queue(
        self,
        rec: sr.Recognizer,
        start_silence: float = START_SILENCE,
        phrase_time_limit: float = None,
    ) -> AudioData:
        """
        Speech recognition on a data stream using queue.
        Queue contains binary buffers where each buffer has raw audio data.

        Args:
            rec: Speech recognizer
            start_silence: Duration of ambient noise calibration
            phrase_time_limit: Maximum recording duration
        """
        # Step 1: adjust to noise level
        # Assumes speech is preceded by at least <startSilence> seconds silence. Loops through this interval
        # to adjust an energy threshold that will subsequently be used to detect speech start.

        # Total time of adjusting noise level
        elapsed_time = 0.0

        # Adjust ambient noise level
        while elapsed_time < start_silence:
            buffer, sound_duration, energy = self._get_next_buffer()
            self._adjust_energy_level(rec, sound_duration, energy)
            elapsed_time += sound_duration

        try_node_logger("Say something (using HSR microphone)!", self.context)

        frames = collections.deque()

        # Step 2: wait for speech to begin
        # beep.SoundRequestPublisher().publish_sound_request()
        # If the energy level exceeds the threshold, consider speech started

        # Wait to speech to begin
        frame_time = self._speech_beginning(rec, frames)

        # At this step, frames contains a list of buffers, and the length of time these buffers recorded is given in
        # frameTime. At this moment, speech should just begun, nonetheless some initial silence is good to keep.
        # Step 3: keep adding to the recorded speech until a long enough pause is detected.

        # Record until pause detected
        self._detect_pause(rec, frames, frame_time, phrase_time_limit)

        return self._convert_to_bytestring(frames)

    def _detect_pause(
        self,
        rec: sr.Recognizer,
        frames: collections.deque,
        frame_time: float,
        phrase_time_limit: float,
    ) -> None:
        pause_time = 0.0

        while True:
            buffer, sound_duration, energy = self._get_next_buffer()
            frames.append((sound_duration, buffer))
            frame_time += sound_duration

            # handle phrase being too long by cutting off the audio
            if phrase_time_limit and frame_time > phrase_time_limit:
                break
            # check if speaking has stopped for longer than the pause threshold on the audio input
            if energy > rec.energy_threshold:
                pause_time = 0.0
            else:
                pause_time += sound_duration

            if pause_time > rec.pause_threshold:
                break

    def _speech_beginning(self, rec: sr.Recognizer, frames: collections.deque) -> float:
        frame_time = 0.0

        while True:
            buffer, sound_duration, energy = self._get_next_buffer()
            frames.append((sound_duration, buffer))
            frame_time += sound_duration

            # Wait until energy exceeds threshold, indicating speech
            if energy > rec.energy_threshold:
                break

            while frame_time > rec.non_speaking_duration:
                d, _ = frames.popleft()
                # remove old frames
                frame_time -= d

            if rec.dynamic_energy_threshold:
                self._adjust_energy_level(rec, sound_duration, energy)

        return frame_time

    @staticmethod
    def _convert_to_bytestring(frames) -> AudioData:
        # Concatenate all the audio frames in frames into a single byte string called frame_data.
        frame_data = b"".join([x[1] for x in frames])
        # Convert the now byte string "frame_data" into a NumPy array called frame_data_array.
        frame_data_array = np.frombuffer(frame_data, dtype=np.int16)
        # Use the noisereduce to apply noise reduction on the audio data stored in frame_data_array.
        noise_reduced_data = nr.reduce_noise(y=frame_data_array, sr=SAMPLE_RATE)
        # Convert the cleaned audio data in noise_reduced_data back into a byte string format, frame_data_clean.
        frame_data_clean = noise_reduced_data.tobytes()
        # Wrap frame_data_clean in an AudioData object from the speech_recognition library.
        return sr.AudioData(frame_data_clean, SAMPLE_RATE, SAMPLE_WIDTH)

    def record_hsr(self, msg: AudioMsg) -> None:
        """
        Callback function for the /audio/audio subscriber to use HSR's microphone for recording.
        Accumulates a numpy array with the recieved AudioData and puts it into a queue which
        gets processed by whisper an rasa.

        Args:
            msg: AudioData recieved from ros topic /audio/audio
            context: a dictionary containing several flags and useful variables
                queue: queue to store the audio data.
                data: accumulated audio data from HSR's microphone.
                lock: lock to ensure that the record_hsr callback does not interfere with the record callback.
                transcriber: thread to perform sound to text
        """
        with self.context.lock:
            if self.context.listening:
                # Normalize incoming message payloads to raw bytes first (works for bytes, lists of ints, memoryviews, etc.)
                try:
                    raw = bytes(msg.data)
                except Exception:
                    # fallback: try converting elements to ints then to bytes
                    raw = bytes([int(x) & 0xFF for x in msg.data])
                # accumulating raw data in numpy array (16-bit little-endian signed samples expected)
                self.context.data = np.concatenate(
                    [self.context.data, np.frombuffer(raw, dtype=np.int16)]
                )

                """
                Checks if the length of context["data"] has at least 32,000 samples. Why do we do this?
                Assuming a sample rate of 16,000 Hz, 32,000 samples would represent 2 seconds of audio. By waiting until the 
                data array has 2 seconds' worth of audio, the we accumulate enough data to process.     
                """
                if len(self.context.data) >= CHUNK_SIZE:
                    # We extract the first 16.000 points of data to use as reference for noisereduction.
                    noise_sample = self.context.data[:SAMPLE_RATE]
                    # Uses the noise sample to remove backround noise from the entire data.
                    reduced_noise_data = nr.reduce_noise(
                        y=self.context.data, sr=SAMPLE_RATE, y_noise=noise_sample
                    )

                    # Adds the reduced_noise_data in the context["queue"].
                    self.context.queue.put(reduced_noise_data)
                    # Reset the array to be empty.
                    self.context.data = np.array([], dtype=np.int16)

    @staticmethod
    def _audio_data_to_numpy(
        audio_data: AudioData, target_sr: int = SAMPLE_RATE
    ) -> tuple[np.ndarray, int]:
        """
        Convert speech_recognition.AudioData to a NumPy float32 waveform.

        Args:
            audio_data (speech_recognition.AudioData): Audio from `r.listen()`.
            target_sr (int): Target sample rate (default: 16000, common in speech recognition).

        Returns:
            np.ndarray: Audio waveform in float32 format (normalized to [-1, 1]).
            int: Sample rate.
        """
        # Get raw audio data as bytes
        raw_data = audio_data.get_raw_data()

        # Convert to NumPy array (int16)
        audio_array = np.frombuffer(raw_data, dtype=np.int16)

        # Convert to float32 and normalize to [-1, 1]
        waveform = librosa.util.buf_to_float(audio_array, dtype=np.float32)

        # Resample if needed
        if audio_data.sample_rate != target_sr:
            waveform = librosa.resample(
                waveform, orig_sr=audio_data.sample_rate, target_sr=target_sr
            )

        return waveform, target_sr

    def _get_next_buffer(self) -> tuple[bytes, float, int]:
        buffer = self.context.queue.get()
        self.context.queue.task_done()
        # Duration of audio buffer in sec
        sound_duration = float(len(buffer) / SAMPLE_RATE)
        energy = audioop.rms(buffer, SAMPLE_WIDTH)

        return buffer, sound_duration, energy

    @staticmethod
    def _adjust_energy_level(
        rec: sr.Recognizer, sound_duration: float, energy: int
    ) -> None:
        damping = rec.dynamic_energy_adjustment_damping**sound_duration
        target_energy = energy * rec.dynamic_energy_ratio
        rec.energy_threshold = rec.energy_threshold * damping + target_energy * (
            1 - damping
        )


class MCRSNode(Node):
    """
    Multi Challenge Robot Script (MCRS) ROS2 node that handles the NLP-Pipeline for all challenges.
    """

    def __init__(self, args):
        super().__init__("MCRS")

        self.get_logger().info("[ALP]: NLP node initialized")

        self._setup_qos()
        self._setup_publisher(args)
        self._setup_subscriber(args)

        self.ctx = Context(
            node=self,
            pub=self.nlp_publisher,
            stt=self.stt_publisher,
            useHSR=args.useHSR,
            useAudio=(args.useAudio != "./"),
            audio=args.useAudio,
            nluURI=args.nluURI,
        )

        self.nlu = NLU(context=self.ctx)
        self.listener = Audio(context=self.ctx)

        self.get_logger().info("[ALP]: NLP node started")
        self.get_logger().info(f"[WHISPER]: Using {device} with {comp_type}.")

    def _setup_qos(self):
        self.qos = QoSProfile(depth=10)
        self.audio_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

    def _setup_publisher(self, args):
        self.nlp_publisher = self.create_publisher(String, args.outputTopic, self.qos)
        self.stt_publisher = self.create_publisher(
            String, args.speechToTextTopic, self.qos
        )

    def _setup_subscriber(self, args):
        if args.useHSR:
            self.create_subscription(
                AudioMsg, "/audio/audio", self.callback_record_hsr, self.audio_qos
            )
        self.create_subscription(String, "/nlp_test", self.callback_nlp_test, self.qos)
        self.create_subscription(
            String, "/startListener", self.callback_start_listener, self.qos
        )

    def callback_record_hsr(self, msg):
        self.listener.record_hsr(msg)

    def callback_nlp_test(self, msg):
        self.nlu.nlu_internal(msg.data, "./")

    def callback_start_listener(self, msg):
        self.listener.start_listener(msg)


def main():
    # Initialize ROS2 and create a node
    rclpy.init(args=sys.argv)

    # Parse command line arguments
    parser = ArgumentParser(prog="activate_language_processing")
    parser.add_argument(
        "-hsr",
        "--useHSR",
        action="store_true",
        help="Flag to record from HSR microphone via the audio capture topic. If you prefer to use the laptop microphone, or directly connect to the microphone instead, do not set this flag.",
    )
    parser.add_argument(
        "-a",
        "--useAudio",
        default="./",
        help="Use an audio file instead of a microphone.Takes the path to an audio file as argument.",
    )
    parser.add_argument(
        "-nlu",
        "--nluURI",
        default="http://localhost:5005/model/parse",
        help="Link towards the RASA semantic parser. Default: http://localhost:5005/model/parse",
    )
    parser.add_argument(
        "-i",
        "--inputTopic",
        default="/nlp_test",
        help="Topic to send texts for the semantic parser, useful for debugging that part of the pipeline. Default: /nlp_test",
    )
    parser.add_argument(
        "-o",
        "--outputTopic",
        default="/nlp_out",
        help="Topic to send semantic parsing results on. Default: /nlp_out",
    )
    parser.add_argument(
        "-stt",
        "--speechToTextTopic",
        default="whisper_out",
        help="Topic to output whisper speech-to-text results on. Default: /whisper_out",
    )
    parser.add_argument(
        "-t",
        "--terminal",
        action="store_true",
        help="Obsolete, this parameter will be ignored: will ALWAYS listen to the input topic.",
    )
    args, unknown = parser.parse_known_args(sys.argv[1:])

    node = MCRSNode(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if "__main__" == __name__:
    main()
