import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import gymnasium as gym
import torch
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from ..utils import Config

gen_config = Config().get_config()


def luna_lander_v2():
    # First, we create our environment called LunarLander-v2
    env = gym.make("LunarLander-v2")

    # Then we reset this environment
    logger.debug("Environment is reset")
    observation, info = env.reset()

    print("_____OBSERVATION SPACE_____ \n")
    """
        Horizontal pad coordinate (x)
        Vertical pad coordinate (y)
        Horizontal speed (x)
        Vertical speed (y)
        Angle
        Angular speed
        If the left leg contact point has touched the land (boolean)
        If the right leg contact point has touched the land (boolean)
    """
    print("Observation Space Shape", env.observation_space.shape)
    print("Sample observation", env.observation_space.sample())  # Get a random observation

    print("\n _____ACTION SPACE_____ \n")
    """
        Action 0: Do nothing,
        Action 1: Fire left orientation engine,
        Action 2: Fire the main engine,
        Action 3: Fire right orientation engine.
    """
    print("Action Space Shape", env.action_space.n)
    print("Action Space Sample", env.action_space.sample())  # Take a random action

    for _ in range(20):
        # Take a random action
        action = env.action_space.sample()
        logger.debug(f"Action taken: {action}")

        # Do this action in the environment and get
        # next_state, reward, terminated, truncated and info
        observation, reward, terminated, truncated, info = env.step(action)
        logger.debug(f"Obs: {observation}, reward: {reward}")

    # If the game is terminated (in our case we land, crashed) or truncated (timeout)
    if terminated or truncated:
        # Reset the environment
        logger.debug("Environment is reset")
        observation, info = env.reset()

    env.close()


def luna_lander_v2_model_train(use_vec_env=False, n_envs=4):
    """
    Train PPO model on LunarLander-v2 with optional GPU and vectorized environments.

    Args:
        use_vec_env: Whether to use vectorized environments for parallel training
        n_envs: Number of parallel environments (only used if use_vec_env=True)
    """
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Create environment (vectorized or single)
    if use_vec_env:
        logger.info(f"Creating {n_envs} parallel environments")
        env = make_vec_env("LunarLander-v2", n_envs=n_envs)
    else:
        env = gym.make("LunarLander-v2")

    # We added some parameters to accelerate the training
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1,
        device=device,  # Use GPU if available
    )

    # Train it for 1,000,000 timesteps
    model.learn(total_timesteps=1000000)

    # Save the model

    # 创建 outputs 目录
    output_dir = Path(gen_config["outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = output_dir / f"ppo-LunarLander-v2_{timestamp}"
    model.save(model_name)
    logger.info(f"Model saved to {model_name}")


def luna_lander_v2_model_eva(model_path=None):
    """
    Evaluate trained PPO model on LunarLander-v2.

    Args:
        model_path: Path to the trained model. If None, uses the latest model in outputs directory.
    """
    if model_path is None:
        # Find the latest model in outputs directory
        output_dir = Path(gen_config["outputs"])
        model_files = list(output_dir.glob("ppo-LunarLander-v2_*.zip"))
        if not model_files:
            logger.error("No trained model found in outputs directory")
            return
        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using latest model: {model_path}")

    model = PPO.load(model_path)
    eval_env = Monitor(gym.make("LunarLander-v2"))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    logger.info(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


def luna_lander_v2_model_play(model_path=None, n_episodes=5, render=True, record_video=False):
    """
    Run trained PPO model on LunarLander-v2 and visualize the gameplay.

    Args:
        model_path: Path to the trained model. If None, uses the latest model in outputs directory.
        n_episodes: Number of episodes to play
        render: Whether to render the environment (requires display)
        record_video: Whether to record video of the gameplay
    """
    if model_path is None:
        # Find the latest model in outputs directory
        output_dir = Path(gen_config["outputs"])
        model_files = list(output_dir.glob("ppo-LunarLander-v2_*.zip"))
        if not model_files:
            logger.error("No trained model found in outputs directory")
            return
        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using latest model: {model_path}")

    # Load the trained model
    model = PPO.load(model_path)
    logger.info(f"Model loaded from {model_path}")

    # Create environment with rendering
    render_mode = "human" if render else None
    if record_video:
        from datetime import datetime

        video_folder = Path(gen_config["outputs"]) / "videos"
        video_folder.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env = gym.make("LunarLander-v2", render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, str(video_folder), name_prefix=f"lunar_lander_{timestamp}")
        logger.info(f"Recording video to {video_folder}")
    else:
        env = gym.make("LunarLander-v2", render_mode=render_mode)

    # Play episodes
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        logger.info(f"Starting episode {episode + 1}/{n_episodes}")

        while not done:
            # Use the trained model to predict action
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        logger.info(f"Episode {episode + 1} finished: Total reward = {total_reward:.2f}, Steps = {steps}")

    env.close()
    logger.info("Gameplay completed")


def luna_lander_v2_model_deploy_api():
    """
    Deploy trained PPO model as a REST API service.
    This function sets up a simple HTTP server that accepts observations and returns actions.
    """

    # Find and load the latest model
    output_dir = Path(gen_config["outputs"])
    model_files = list(output_dir.glob("ppo-LunarLander-v2_*.zip"))
    if not model_files:
        logger.error("No trained model found in outputs directory")
        return
    model_path = max(model_files, key=lambda p: p.stat().st_mtime)

    model = PPO.load(model_path)
    logger.info(f"Model loaded from {model_path}")

    class ModelHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/predict":
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode("utf-8"))

                # Get observation from request
                observation = data.get("observation")
                if observation is None:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error": "observation is required"}')
                    return

                # Predict action
                action, _states = model.predict(observation, deterministic=True)

                # Return action
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                response = {"action": int(action)}
                self.wfile.write(json.dumps(response).encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            logger.info(f"{self.address_string()} - {format % args}")

    # Start HTTP server
    port = gen_config.get("http", {}).get("port", 13000)
    server = HTTPServer(("0.0.0.0", port), ModelHandler)
    logger.info(f"Model API server started on port {port}")
    logger.info(f"Send POST requests to http://localhost:{port}/predict with JSON body: {{'observation': [...]}}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.error("Shutting down server...")
        server.shutdown()
