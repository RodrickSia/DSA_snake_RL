import pygame
import numpy as np
import random
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import glob
import cv2


# ====================== CONFIGURATION ======================
WIDTH, HEIGHT = 800, 600           # Window size
BLOCK = 20                         # Grid block size (px)
SPEED = 40                         # FPS when rendering

# Q‑learning hyper‑parameters
ALPHA = 0.1       # learning rate
GAMMA = 0.9       # discount factor
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

EPISODES = 2500   # total training episodes
MAX_STEPS = 2000   # max steps per episode

# Directory setup
RUN_DIR = "runs"
LOG_DIR = "logs"
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

FRAMES_DIR = os.path.join(RUN_DIR, "frames_q_learning")
os.makedirs(FRAMES_DIR, exist_ok=True)

# Configure logging
log_filename = os.path.join(LOG_DIR, f"snake_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.info("==== Training started ====")

# ====================== ENVIRONMENT ========================
class SnakeGame:
    def __init__(self, render: bool = False):
        self.w, self.h = WIDTH, HEIGHT
        self.block = BLOCK
        self.grid_w = self.w // self.block
        self.grid_h = self.h // self.block
        self.render_mode = render
        if self.render_mode:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Snake Q‑Learning")
            self.clock = pygame.time.Clock()
        self.reset()

    # Reset environment
    def reset(self):
        self.direction = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])  # U/D/L/R
        x = self.grid_w // 2
        y = self.grid_h // 2
        self.snake = [(x, y), (x - self.direction[0], y - self.direction[1])]
        self.place_food()
        self.frame = 0
        return self.get_state()

    # Place food randomly not on snake
    def place_food(self):
        while True:
            fx = random.randint(0, self.grid_w - 1)
            fy = random.randint(0, self.grid_h - 1)
            if (fx, fy) not in self.snake:
                self.food = (fx, fy)
                break

    # Get encoded state (11‑bit tuple)
    def get_state(self):
        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.direction
        # Danger detection helpers
        def danger_in_direction(dx, dy):
            nx, ny = head_x + dx, head_y + dy
            if nx < 0 or nx >= self.grid_w or ny < 0 or ny >= self.grid_h:
                return 1
            if (nx, ny) in self.snake:
                return 1
            return 0

        # straight, right, left relative to current direction
        danger_straight = danger_in_direction(dir_x, dir_y)
        danger_right = danger_in_direction(dir_y, -dir_x)
        danger_left = danger_in_direction(-dir_y, dir_x)

        # Direction encoding
        dir_up = 1 if self.direction == (0, -1) else 0
        dir_down = 1 if self.direction == (0, 1) else 0
        dir_left = 1 if self.direction == (-1, 0) else 0
        dir_right = 1 if self.direction == (1, 0) else 0

        # Food relative position
        food_up = 1 if self.food[1] < head_y else 0
        food_down = 1 if self.food[1] > head_y else 0
        food_left = 1 if self.food[0] < head_x else 0
        food_right = 1 if self.food[0] > head_x else 0

        state = (
            danger_straight,
            danger_right,
            danger_left,
            dir_up,
            dir_down,
            dir_left,
            dir_right,
            food_up,
            food_down,
            food_left,
            food_right,
        )
        return state

    # State -> integer index
    @staticmethod
    def state_to_int(state):
        idx = 0
        for bit in state:
            idx = (idx << 1) | bit
        return idx

    # Perform action: 0 = straight, 1 = right turn, 2 = left turn
    def step(self, action):
        self.frame += 1
        # Update direction based on action
        dir_x, dir_y = self.direction
        if action == 1:  # right turn
            self.direction = (dir_y, -dir_x)
        elif action == 2:  # left turn
            self.direction = (-dir_y, dir_x)
        dir_x, dir_y = self.direction

        # Move snake
        head_x, head_y = self.snake[0]
        new_head = (head_x + dir_x, head_y + dir_y)

        reward = 0
        done = False

        # Collision with wall or self
        if (
            new_head[0] < 0
            or new_head[0] >= self.grid_w
            or new_head[1] < 0
            or new_head[1] >= self.grid_h
            or new_head in self.snake
        ):
            done = True
            reward = -10
            return self.get_state(), reward, done

        # Insert new head
        self.snake.insert(0, new_head)

        # Food eaten
        if new_head == self.food:
            reward = 10
            self.place_food()
        else:
            self.snake.pop()  # remove tail if no food

        # Small survival reward to encourage longer games
        reward += 0.1

        if self.frame > MAX_STEPS:
            done = True
        return self.get_state(), reward, done

    # Render and optionally save frame
    def render(self, save_path: str | None = None):
        if not self.render_mode:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # Draw background
        self.display.fill((0, 0, 0))
        # Draw food
        pygame.draw.rect(
            self.display,
            (255, 0, 0),
            pygame.Rect(self.food[0] * self.block, self.food[1] * self.block, self.block, self.block),
        )
        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            color = (0, 255, 0) if i == 0 else (0, 200, 0)
            pygame.draw.rect(
                self.display,
                color,
                pygame.Rect(x * self.block, y * self.block, self.block, self.block),
            )

        # Draw snake length in top-left corner
        font = pygame.font.SysFont(None, 24)
        text = font.render(f'length snake= {len(self.snake)}', True, (255, 255, 255))
        self.display.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(SPEED)
        if save_path is not None:
            pygame.image.save(self.display, save_path)

# ====================== Q‑LEARNING AGENT ====================
class QAgent:
    def __init__(self, state_size: int, action_size: int = 3):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size), dtype=np.float32)
        self.epsilon = EPSILON_START

    def act(self, state_int):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return int(np.argmax(self.q_table[state_int]))

    def learn(self, s_int, a, r, s_next_int, done):
        target = r + (0 if done else GAMMA * np.max(self.q_table[s_next_int]))
        self.q_table[s_int, a] += ALPHA * (target - self.q_table[s_int, a])

    def decay_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

# ====================== TRAINING LOOP =======================

def train():
    rewards = []  
    env = SnakeGame(render=False)
    n_states = 2 ** 11  # 11‑bit state space
    agent = QAgent(n_states)
    best_score = 0
    loop_threshold = 50  # if same state is visited > 50 times in an episode, break the loop
    for ep in range(1, EPISODES + 1):
        state = env.reset()
        s_int = env.state_to_int(state)
        total_reward = 0
        steps = 0
        state_visit = {}  # loop-detection counter
        while True:
            action = agent.act(s_int)
            state_next, reward, done = env.step(action)
            s_next_int = env.state_to_int(state_next)
            agent.learn(s_int, action, reward, s_next_int, done)
            s_int = s_next_int
            total_reward += reward
            steps += 1

            # Update loop counter:
            state_visit[s_int] = state_visit.get(s_int, 0) + 1
            if state_visit[s_int] > loop_threshold:
                # Force break if the same state repeats too many times
                logging.info(f"Loop detected at EP {ep:04d} after {steps} steps.")
                break

            if done:
                break
            
        rewards.append(total_reward)
        agent.decay_epsilon()
        # Log episode results
        logging.info(
            f"EP {ep:04d} | steps: {steps:03d} | reward: {total_reward:.1f} | eps: {agent.epsilon:.3f} | len: {len(env.snake)}"
        )
        if len(env.snake) > best_score:
            best_score = len(env.snake)
            np.save(os.path.join(LOG_DIR, "best_qtable.npy"), agent.q_table)
        # Render & save every 100 episodes
        if ep % 100 == 0 or ep == EPISODES:
            save_episode(env, agent, ep)
    # Save final Q‑table
    np.save(os.path.join(LOG_DIR, "qtable_final.npy"), agent.q_table)
    logging.info("==== Training finished ====")

      # === Plot the reward chart ===
    window = 50  # Number of episodes to calculate moving average
    def moving_average(x, w):
        return np.convolve(x, np.ones(w)/w, mode='valid')

    smoothed = moving_average(rewards, window)
    episodes_ma = range(window-1, len(rewards))

    plt.figure(figsize=(10,5))
    plt.plot(rewards, alpha=0.3, label='Raw Reward')
    plt.plot(episodes_ma, smoothed, label=f'MA (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward per Episode')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ====================== SAVE EPISODE PLAY ===================

def save_episode(env: SnakeGame, agent: QAgent, episode: int):
    folder = os.path.join(FRAMES_DIR, f"ep_{episode:04d}")
    os.makedirs(folder, exist_ok=True)
    
    # Store original render mode
    original_render_mode = env.render_mode
    env.render_mode = True
    
    # Initialize pygame components if they don't exist
    if not pygame.get_init():
        pygame.init()
    
    # Create display and clock if they weren't initialized before
    if not hasattr(env, 'display') or env.display is None:
        env.display = pygame.display.set_mode((env.w, env.h))
        pygame.display.set_caption("Snake Q‑Learning")
    
    if not hasattr(env, 'clock') or env.clock is None:
        env.clock = pygame.time.Clock()
    
    # Use the learned policy without exploration for visualization
    saved_epsilon = agent.epsilon
    agent.epsilon = 0
    
    state = env.reset()
    s_int = env.state_to_int(state)
    step = 0
    while True:
        action = agent.act(s_int)
        state_next, _, done = env.step(action)
        s_int = env.state_to_int(state_next)
        frame_path = os.path.join(folder, f"frame_{step:04d}.png")
        env.render(save_path=frame_path)
        step += 1
        if done or step >= MAX_STEPS:
            break
    
        # === Compile frames into MP4 video ===
    video_path = os.path.join(folder, f"ep_{episode:04d}.mp4")
    frame_files = sorted(glob.glob(os.path.join(folder, "frame_*.png")))
    if frame_files:
        img0 = cv2.imread(frame_files[0])
        h, w, _ = img0.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(video_path, fourcc, SPEED, (w, h))
        for f in frame_files:
            img = cv2.imread(f)
            video.write(img)
        video.release()
        logging.info(f"Episode {episode:04d} video saved to {video_path}")

    # Restore original settings
    agent.epsilon = saved_epsilon
    # pygame.quit()
    env.render_mode = original_render_mode
    
    logging.info(f"Episode {episode} saved to {folder}")

# ====================== MAIN ================================
if __name__ == "__main__":
    train()
