import pickle
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from tqdm import tqdm


class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        current_q = self.q_values[obs][action]
        temporal_difference = target - current_q  # error

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[obs][action] = current_q + self.lr * temporal_difference

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


def train_agent():
    # Training hyperparameters
    learning_rate = 0.01  # How fast to learn (higher = faster but less stable)
    n_episodes = 100_000  # Number of hands to practice
    start_epsilon = 1.0  # Start with 100% random actions
    epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
    final_epsilon = 0.1  # Always keep some exploration

    # Create environment and agent
    env = gym.make("Blackjack-v1", sab=False)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    agent = BlackjackAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    logger.debug("Start training agent")
    for _episode in tqdm(range(n_episodes)):
        # Start a new hand
        obs, info = env.reset()
        done = False

        # Play one complete hand
        while not done:
            # Agent chooses action (initially random, gradually more intelligent)
            action = agent.get_action(obs)

            # Take action and observe result
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Learn from this experience
            agent.update(obs, action, reward, terminated, next_obs)

            # Move to next state
            done = terminated or truncated
            obs = next_obs

        # Reduce exploration rate (agent becomes less random over time)
        agent.decay_epsilon()

    # Save the trained model to outputs directory
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    model_path = outputs_dir / "blackjack_agent.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "q_values": dict(agent.q_values),
                "training_error": agent.training_error,
                "epsilon": agent.epsilon,
                "learning_rate": agent.lr,
                "discount_factor": agent.discount_factor,
            },
            f,
        )

    logger.debug(f"Model saved to {model_path}")
    logger.debug(f"Final epsilon: {agent.epsilon:.4f}")
    logger.debug(f"Q-table size: {len(agent.q_values)} states")


def load_agent(model_path: str = "outputs/blackjack_agent.pkl") -> BlackjackAgent:
    """Load a trained agent from a saved model file.

    Args:
        model_path: Path to the saved model file

    Returns:
        BlackjackAgent: Loaded agent with trained Q-values
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    # Create environment
    env = gym.make("Blackjack-v1", sab=False)

    # Create agent with saved hyperparameters
    agent = BlackjackAgent(
        env=env,
        learning_rate=model_data["learning_rate"],
        initial_epsilon=model_data["epsilon"],
        epsilon_decay=0.0,
        final_epsilon=model_data["epsilon"],
        discount_factor=model_data["discount_factor"],
    )

    # Load trained Q-values
    agent.q_values = defaultdict(lambda: np.zeros(env.action_space.n), model_data["q_values"])
    agent.training_error = model_data["training_error"]

    return agent


def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(np.array(arr).flatten(), np.ones(window), mode=convolution_mode) / window


def analyze_agent(model_path: str = "outputs/blackjack_agent.pkl"):
    """Analyze a trained agent by loading it from file and visualizing training metrics.

    Args:
        model_path: Path to the saved model file
    """
    agent = load_agent(model_path)

    # Smooth over a 500-episode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=1, figsize=(12, 5))

    # Training error (how much we're still learning)
    axs.set_title("Training Error")
    training_error_moving_average = get_moving_avgs(agent.training_error, rolling_length, "same")
    axs.plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs.set_ylabel("Temporal Difference Error")
    axs.set_xlabel("Step")

    plt.tight_layout()
    plt.show()


def eva_agent(model_path: str = "outputs/blackjack_agent.pkl", num_episodes: int = 1000):
    """Evaluate agent performance without learning or exploration.

    Args:
        model_path: Path to the saved model file
        num_episodes: Number of episodes to test
    """
    agent = load_agent(model_path)
    env = agent.env

    total_rewards = []

    # Temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in tqdm(range(num_episodes), desc="Evaluating"):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    # Restore original epsilon
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    logger.debug(f"Test Results over {num_episodes} episodes:")
    logger.debug(f"Win Rate: {win_rate:.1%}")
    logger.debug(f"Average Reward: {average_reward:.3f}")
    logger.debug(f"Standard Deviation: {np.std(total_rewards):.3f}")
