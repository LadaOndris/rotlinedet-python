import argparse
import time

import cv2

from core import process_frame


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
    args = parser.parse_args()

    if args.loops < 1:
        args.loops = 1
    if args.max_frames < 1:
        args.max_frames = 0

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
                          rotation_step: int) -> int:
    print(F"Processing max frames: {max_frames}")
    print(F"Rotation step is: {rotation_step}")

    processed_frames = 0
    while processed_frames != max_frames:
        frame = input_provider.next_frame()
        if frame is None:
            break
        process_frame(frame, rotation_step)
        processed_frames += 1
    return processed_frames


def start_processing(input_provider: InputProvider, max_frames: int,
                     rotation_step: int) -> None:
    start_time = time.time()
    processed_frames = 0

    try:
        processed_frames = start_processing_loop(input_provider, max_frames, rotation_step)
    except KeyboardInterrupt:
        print("Interrupted! Ending ...")

    end_time = time.time()
    print_statistics(start_time, end_time, processed_frames)


if __name__ == "__main__":
    args = parse_args()
    input_provider = InputProvider(args.input, args.loops)
    start_processing(input_provider, args.max_frames, args.rotation_step)
