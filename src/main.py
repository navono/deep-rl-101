import argparse
import asyncio

from .app import start


def parse_args():
    parser = argparse.ArgumentParser(description="Deep RL 101 - Run various functions")
    parser.add_argument(
        "func",
        nargs="?",
        default="eva",
        choices=["luna", "train", "eva", "llm", "img"],
        help="Function to run: luna (luna_lander_v2), train (luna_lander_v2_model_train), eva (luna_lander_v2_model_eva), llm (llm_generate), img (img_generation)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(start(args.func))
