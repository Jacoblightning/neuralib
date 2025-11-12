#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pillow",
# ]
# ///


import sys
import os.path

from PIL import Image

def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <image>")
        return 1

    if not os.path.exists(sys.argv[1]):
        print(f"Error: {sys.argv[1]} not found.")
        return 1

    image = Image.open(sys.argv[1])

    print([float(i) for i in image.tobytes()])

    return 0

if __name__ == "__main__":
    exit(main())
