import argparse
import cProfile
import io
import pstats
import time

import cv2

from core import process_frame

profile = cProfile.Profile()


def parse_args():
    parser = argparse.ArgumentParser(description='Adaptive contrast enhancement using WTHE method.')
    parser.add_argument('-i', metavar='PATH', dest='input', action='store', default=None,
                        help='input file with source image or video (default: use input from camera)')
    parser.add_argument('-l', metavar='L', type=int, dest='loops', action='store', default=1,
                        help='loop input image or video N times (default: 1), 0 means forever')
    parser.add_argument('-c', metavar='C', type=int, dest='max_frames', action='store', default=0,
                        help='stop processing prematurely after N frames (default: 0), 0 means no limit')
    parser.add_argument('-r', metavar='R', type=float, dest='rotation_step', action='store', default=1,
                        help='rotation step in degrees (default: 1), 1 means 180 rotations')
    parser.add_argument('-p', '--profile', dest='is_profiling_enabled', action='store_true')
    args = parser.parse_args()

    if args.loops < 1:
        args.loops = 1
    if args.max_frames < 1:
        args.max_frames = 1

    return args


class InputProvider:

    def __init__(self, input_path: str, loops: int):
        self.input_path = input_path
        self.loops = loops

        self.capture = None
        self.prepare_input()

    def prepare_input(self):
        # Input and output preparation
        print("Preparing input and output ...")
        if self.input_path is None:
            self.capture = cv2.VideoCapture(0)
        else:
            self.capture = cv2.VideoCapture(self.input_path)
        if not self.capture.isOpened():
            exit("ERROR: Unable to read input data!")

    def next_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            # no more frames -> input is at the end, repeat
            self.loops -= 1
            if self.loops != 0:  # restart the processing
                self.prepare_input()
                return self.next_frame()
        return frame


def print_statistics(start_time, end_time, frames):
    processing_time = end_time - start_time
    processing_speed = frames / processing_time
    print()
    print(f"Processing time: {processing_time:.3f} s")
    print(f"Processed frames: {frames}")
    print(f"Processing speed: {processing_speed:.3f} FPS")


def start_processing_loop(input_provider: InputProvider, max_frames: int,
                          rotation_step: int, is_profiling_enabled: bool) -> int:
    print(F"Processing max frames: {max_frames}")
    print(F"Rotation step is: {rotation_step}")

    processed_frames = 0
    while processed_frames != max_frames:
        frame = input_provider.next_frame()
        if frame is None:
            break

        if is_profiling_enabled:
            profile.enable()
        process_frame(frame, rotation_step)
        if is_profiling_enabled:
            profile.disable()

        processed_frames += 1
    return processed_frames


def start_processing(input_provider: InputProvider, max_frames: int,
                     rotation_step: int, is_profiling_enabled: bool) -> None:
    start_time = time.time()
    processed_frames = 0

    try:
        processed_frames = start_processing_loop(input_provider, max_frames, rotation_step, is_profiling_enabled)
    except KeyboardInterrupt:
        print("Interrupted! Ending ...")

    end_time = time.time()
    print_statistics(start_time, end_time, processed_frames)


def print_profiling_result():
    s = io.StringIO()
    ps = pstats.Stats(profile, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())


if __name__ == "__main__":
    arguments = parse_args()
    provider = InputProvider(arguments.input, arguments.loops)

    start_processing(provider, arguments.max_frames, arguments.rotation_step, arguments.is_profiling_enabled)

    if arguments.is_profiling_enabled:
        print_profiling_result()
