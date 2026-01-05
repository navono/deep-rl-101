from src.hf_rl_class.unit1 import luna_lander_v2_model_train

from .utils import Config, CustomizeLogger

gen_config = Config().get_config()
logger = CustomizeLogger.make_logger(gen_config["log"])


async def start():
    logger.info("Hello from deep RL 101!")

    # llm_generate()
    # img_generation()

    # luna_lander_v2()
    luna_lander_v2_model_train()
