import os
import numpy as np
from prettytable import PrettyTable
from snake_q_learning import SnakeGame, QAgent, MAX_STEPS, LOG_DIR  # Import necessary components

def evaluate(num_episodes=10):
    # Load final Q‑table from file
    qtable_path = os.path.join(LOG_DIR, "qtable_final.npy")
    if not os.path.exists(qtable_path):
        print(f"Q‑table file not found at {qtable_path}")
        return []
    q_table = np.load(qtable_path)
    
    # Create agent with greedy policy (epsilon = 0)
    state_size = 2 ** 11
    agent = QAgent(state_size)
    agent.q_table = q_table
    agent.epsilon = 0  # turn off exploration for evaluation
    
    results = []
    for ep in range(1, num_episodes + 1):
        env = SnakeGame(render=False)
        state = env.reset()
        s_int = env.state_to_int(state)
        total_reward = 0
        steps = 0

        while True:
            action = agent.act(s_int)
            state, reward, done = env.step(action)
            s_int = env.state_to_int(state)
            total_reward += reward
            steps += 1
            if done or steps >= MAX_STEPS:
                break

        results.append((ep, steps, total_reward, len(env.snake)))
    return results

def main():
    num_episodes = 200  # Set how many evaluation episodes you want to run
    metrics = evaluate(num_episodes)
    
    # Create a table to display metrics using PrettyTable
    table = PrettyTable(["Episode", "Steps", "Total Reward", "Final Snake Length"])
    for ep, steps, reward, length in metrics:
        table.add_row([ep, steps, f"{reward:.1f}", length])
    
    print(table)

if __name__ == "__main__":
    main()