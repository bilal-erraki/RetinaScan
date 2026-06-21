"""Predict one local image from the command line."""

import argparse
import json

from src.predict import predict_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a retinal fundus image.")
    parser.add_argument("image", help="Path to a PNG or JPEG retinal image")
    args = parser.parse_args()
    print(json.dumps(predict_image(args.image), indent=2))


if __name__ == "__main__":
    main()
