from src.hf_rl_class.luna_lander import luna_lander_v2, luna_lander_v2_model_deploy_api, luna_lander_v2_model_eva, luna_lander_v2_model_play, luna_lander_v2_model_train
from src.utils.llm import img_generate, llm_generate

from .utils import Config, CustomizeLogger

gen_config = Config().get_config()
logger = CustomizeLogger.make_logger(gen_config["log"])


async def start(func: str = "eva"):
    logger.info(f"Hello from deep RL 101! Running function: {func}")

    func_map = {
        "luna": luna_lander_v2,
        "train": luna_lander_v2_model_train,
        "eva": luna_lander_v2_model_eva,
        "play": luna_lander_v2_model_play,
        "api": luna_lander_v2_model_deploy_api,
        "llm": llm_generate,
        "img": img_generate,
    }

    if func not in func_map:
        logger.error(f"Unknown function: {func}")
        logger.info(f"Available functions: {', '.join(func_map.keys())}")
        return

    target_func = func_map[func]
    if target_func is None:
        logger.warning(f"Function '{func}' is not yet implemented")
        return

    target_func()
