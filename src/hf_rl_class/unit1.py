import gymnasium as gym
import torch
from loguru import logger
from stable_baselines3 import PPO


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


def luna_lander_v2_model_train():
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Create environment
    env = gym.make("LunarLander-v2")

    # # Instantiate the agent
    # model = PPO("MlpPolicy", env, verbose=1)
    # # Train the agent
    # model.learn(total_timesteps=int(2e5))

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
    model_name = "ppo-LunarLander-v2"
    model.save(model_name)
    logger.info(f"Model saved to {model_name}")
