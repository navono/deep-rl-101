from src.gym_learning.basic_rl import first_rl, rl_spaces
from src.gym_learning.training_agent import analyze_agent as blackjack_analyze_agent
from src.gym_learning.training_agent import eva_agent as blackjack_eva_agent
from src.gym_learning.training_agent import train_agent as blackjack_train_agent
from src.hf_rl_class.unit_0_luna_lander import luna_lander_v2, luna_lander_v2_model_deploy_api, luna_lander_v2_model_eva, luna_lander_v2_model_play, luna_lander_v2_model_train
from src.utils.llm import img_generate, llm_generate

from .utils import Config, CustomizeLogger

gen_config = Config().get_config()
logger = CustomizeLogger.make_logger(gen_config["log"])


async def start(func: str = "eva"):
    logger.info(f"Hello from deep RL 101! Running function: {func}")

    func_map = {
        "llm": llm_generate,
        "img": img_generate,
        "unit_0_luna": luna_lander_v2,
        "unit_0_train": luna_lander_v2_model_train,
        "unit_0_eva": luna_lander_v2_model_eva,
        "unit_0_play": luna_lander_v2_model_play,
        "unit_0_api": luna_lander_v2_model_deploy_api,
        "basic_rl": first_rl,
        "rl_spaces": rl_spaces,
        "blackjack_train": blackjack_train_agent,
        "blackjack_eva": blackjack_eva_agent,
        "blackjack_analyze": blackjack_analyze_agent,
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
