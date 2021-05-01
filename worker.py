import os
import time
import sys
import argparse

parser = argparse.ArgumentParser(description='Worker')
parser.add_argument('device', type=str, default="0")

args, _ = parser.parse_known_args()

while True:
    error_code = os.system(f"python runner.py auto --device {args.device}")
    if error_code != 0:
        print("Error code", error_code)
        time.sleep(5 * 60)

    time.sleep(0.5 * 60)
