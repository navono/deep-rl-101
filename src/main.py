import argparse
import asyncio

from .app import start


def parse_args():
    parser = argparse.ArgumentParser(description="Deep RL 101 - Run various functions")
    parser.add_argument(
        "func",
        nargs="?",
        default="eva",
        choices=["llm", "img", "unit_0_luna", "unit_0_train", "unit_0_eva", "unit_0_play", "unit_0_api", "basic_rl"],
        help="Function to run: unit_0_luna (luna_lander_v2), unit_0_train (luna_lander_v2_model_train), unit_0_eva (luna_lander_v2_model_eva), unit_0_play (luna_lander_v2_model_play), unit_0_api (luna_lander_v2_model_deploy_api), llm (llm_generate), img (img_generation)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(start(args.func))
