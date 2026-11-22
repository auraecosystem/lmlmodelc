"""Command-line entry point."""
import argparse
from .pipeline import LMLPipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="lmlmodel")
    sub = parser.add_subparsers(dest="command", required=True)
    info = sub.add_parser("inspect", help="validate and inspect a safe checkpoint")
    info.add_argument("checkpoint")
    args = parser.parse_args()
    if args.command == "inspect":
        pipeline = LMLPipeline.from_pretrained(args.checkpoint, device="cpu")
        print(pipeline.model.config.to_dict())


if __name__ == "__main__":
    main()
