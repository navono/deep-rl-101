# from .local_llm import simple_llm_generate, simple_img_generate
from src.hf_rl_class.unit1 import luna_lander_v2

from .utils import Config, CustomizeLogger

gen_config = Config().get_config()
logger = CustomizeLogger.make_logger(gen_config["log"])


async def start():
    logger.info("Hello from py-project-template!")
    # simple_llm_generate()
    # simple_img_generate()

    luna_lander_v2()
