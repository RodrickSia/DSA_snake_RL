import pygame
import numpy as np
import random
import os
import logging
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt

# ============================================================
# ------------------  CONFIGURATION  -------------------------
# ============================================================
WIDTH, HEIGHT   = 800, 600         # Window size
BLOCK           = 20               # Grid block size (px)
RENDER_FPS      = 60               # FPS when visualising

# ---------- RL hyper-parameters ----------
EPISODES        = 2000
MAX_STEPS       = 750
BATCH_SIZE      = 64
MEMORY_SIZE     = 30000
GAMMA           = 0.99             # discount factor

LR              = 1e-3             # learning-rate for Net
EPS_START       = 1.0
EPS_MIN         = 0.05
EPS_DECAY       = 0.995
TARGET_SYNC     = 250              # copy online→target every N updates

# ---------- folders & logging ----------
RUN_DIR = "runs"
LOG_DIR = "logs"
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f"snake_DQN_{datetime.now():%Y%m%d_%H%M%S}.log")
logging.basicConfig(filename=log_filename,
                    level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logging.info("==== DQN training started ====")

# ============================================================
# ------------------  AUTOGRAD CORE  -------------------------
# ============================================================
class Node:
    """
    A *very* small autograd node.
    Holds a tensor `val` and its gradient `grad`.
    Forward pass just moves values through the graph.
    Backward pass accumulates dL/d(param) into `grad`.
    """
    def __init__(self, val: np.ndarray):
        self.val  = val.astype(np.float32, copy=False)
        self.grad = np.zeros_like(self.val, dtype=np.float32)

    def zero_grad(self):
        self.grad.fill(0.)

# ============================================================
# -------------  LAYER (Linear + optional ReLU)  -------------
# ============================================================
class Layer:
    """
    Fully-connected layer:  y = xW + b
    Optionally followed by ReLU.
    During forward we cache `x` so that backward can
    compute gradients w.r.t. W, b *and* propagate dL/dx.
    """
    def __init__(self, in_dim: int, out_dim: int, act_relu: bool = True):
        limit = 1. / np.sqrt(in_dim)  # Xavier/Glorot-uniform range
        self.W = Node(np.random.uniform(-limit, limit, (in_dim, out_dim)))
        self.b = Node(np.zeros(out_dim, dtype=np.float32))
        self.act_relu = act_relu
        # Cache for backpropagation
        self._last_x = None
        self._mask = None  # for ReLU backward

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x : (batch, in_dim)
        returns y : (batch, out_dim)
        """
        self._last_x = x
        y = x @ self.W.val + self.b.val
        if self.act_relu:
            self._mask = (y > 0).astype(np.float32)
            y = y * self._mask
        return y

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        grad_out : dL/dy  (shape: (batch, out_dim))
        returns   : dL/dx  (shape: (batch, in_dim)) to propagate deeper
        """
        if self.act_relu:
            grad_out = grad_out * self._mask

        # Compute gradients for weights and biases
        x = self._last_x  # shape: (batch, in_dim)
        self.W.grad += x.T @ grad_out  # (in_dim, out_dim)
        self.b.grad += grad_out.sum(axis=0)

        # Compute gradient to propagate to previous layer
        return grad_out @ self.W.val.T

# ============================================================
# ----------------  NEURAL NETWORK (DQN)  --------------------
# ============================================================
class Neural:
    """
    A tiny feed-forward network built from Layers.
    Acts as the function approximator for Q(s, a; θ).

    Topology:  (11) → 64 → 64 → 3   (all ReLU except last)
    """
    def __init__(self, input_dim: int, output_dim: int):
        self.layers = [
            Layer(input_dim, 64, act_relu=True),
            Layer(64, 64, act_relu=True),
            Layer(64, output_dim, act_relu=False)  # Last layer linear
        ]
        # Build list of trainable parameters for convenience
        self.params = []
        for lyr in self.layers:
            self.params += [lyr.W, lyr.b]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward compute Q-values.
        x shape = (batch, input_dim)
        """
        for lyr in self.layers:
            x = lyr.forward(x)
        return x

    def backward(self, loss_grad: np.ndarray):
        """
        Propagate gradient from dL/dOut (shape same as network output)
        down to parameters (W, b) in each layer.
        """
        for lyr in reversed(self.layers):
            loss_grad = lyr.backward(loss_grad)

    def step(self, lr: float):
        """
        Update parameters using Gradient Descent and reset gradients.
        """
        for p in self.params:
            p.val -= lr * p.grad
            p.zero_grad()

    def copy_from(self, other: "Neural"):
        """
        Copy parameters from another network.
        """
        for p_t, p_s in zip(self.params, other.params):
            np.copyto(p_t.val, p_s.val)

# ============================================================
# ------------------  SNAKE ENVIRONMENT  ---------------------
# ============================================================
class SnakeGame:
    # Note: Use your existing SnakeGame code. For brevity only changes are shown.
    def __init__(self, render: bool = False):
        self.w, self.h = WIDTH, HEIGHT
        self.block = BLOCK
        self.grid_w = self.w // self.block
        self.grid_h = self.h // self.block
        self.render_mode = render
        if self.render_mode:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Snake Deep-Q-Learning")
            self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
        x = self.grid_w // 2
        y = self.grid_h // 2
        self.snake = [(x, y), (x - self.direction[0], y - self.direction[1])]
        self.place_food()
        self.frame = 0
        return self.get_state()

    def place_food(self):
        while True:
            fx = random.randint(0, self.grid_w - 1)
            fy = random.randint(0, self.grid_h - 1)
            if (fx, fy) not in self.snake:
                self.food = (fx, fy)
                break

    def get_state(self):
        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.direction
        def danger_in_direction(dx, dy):
            nx, ny = head_x + dx, head_y + dy
            if nx < 0 or nx >= self.grid_w or ny < 0 or ny >= self.grid_h:
                return 1
            if (nx, ny) in self.snake:
                return 1
            return 0

        danger_straight = danger_in_direction(dir_x, dir_y)
        danger_right = danger_in_direction(dir_y, -dir_x)
        danger_left = danger_in_direction(-dir_y, dir_x)

        dir_up = 1 if self.direction == (0, -1) else 0
        dir_down = 1 if self.direction == (0, 1) else 0
        dir_left = 1 if self.direction == (-1, 0) else 0
        dir_right = 1 if self.direction == (1, 0) else 0

        food_up = 1 if self.food[1] < head_y else 0
        food_down = 1 if self.food[1] > head_y else 0
        food_left = 1 if self.food[0] < head_x else 0
        food_right = 1 if self.food[0] > head_x else 0

        state = (danger_straight, danger_right, danger_left,
                 dir_up, dir_down, dir_left, dir_right,
                 food_up, food_down, food_left, food_right)
        return state

    @staticmethod
    def state_to_int(state):
        idx = 0
        for bit in state:
            idx = (idx << 1) | bit
        return idx

    def step(self, action):
        self.frame += 1
        dir_x, dir_y = self.direction
        if action == 1:  # right turn
            self.direction = (dir_y, -dir_x)
        elif action == 2:  # left turn
            self.direction = (-dir_y, dir_x)
        dir_x, dir_y = self.direction

        head_x, head_y = self.snake[0]
        new_head = (head_x + dir_x, head_y + dir_y)

        reward = 0
        done = False

        if (new_head[0] < 0 or new_head[0] >= self.grid_w or
            new_head[1] < 0 or new_head[1] >= self.grid_h or
            new_head in self.snake):
            done = True
            reward = -10
            return self.get_state(), reward, done

        self.snake.insert(0, new_head)
        if new_head == self.food:
            reward = 10
            self.place_food()
        else:
            self.snake.pop()

        reward += 0.1
        if self.frame > MAX_STEPS:
            done = True
        return self.get_state(), reward, done

    def render(self, save_path: str = None):
        if not self.render_mode:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self.display.fill((0, 0, 0))
        pygame.draw.rect(self.display, (255, 0, 0),
                         pygame.Rect(self.food[0]*self.block, self.food[1]*self.block, self.block, self.block))
        for i, (x, y) in enumerate(self.snake):
            color = (0, 255, 0) if i == 0 else (0, 200, 0)
            pygame.draw.rect(self.display, color,
                             pygame.Rect(x*self.block, y*self.block, self.block, self.block))
        pygame.display.flip()
        self.clock.tick(RENDER_FPS)
        if save_path is not None:
            pygame.image.save(self.display, save_path)

# ============================================================
# ------------------  REPLAY BUFFER  -------------------------
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(tuple(transition))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buf)

# ============================================================
# -------------------  DQN AGENT  ----------------------------
# ============================================================
class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.online = Neural(state_dim, action_dim)
        self.target = Neural(state_dim, action_dim)

        # ————— Load pretrained weights into the online network —————
        data = np.load("pretrained_qnet.npz")
        for p, key in zip(self.online.params, data.files):
            np.copyto(p.val, data[key])

        self.target.copy_from(self.online)
        print(">> DQNAgent: loaded pretrained_qnet.npz")

        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.eps = EPS_START
        self.action_dim = action_dim
        self.t_step = 0

    def choose_action(self, state_vec: np.ndarray) -> int:
        if random.random() < self.eps:
            return random.randrange(self.action_dim)
        q_vals = self.online(state_vec[None, :])[0]
        return int(np.argmax(q_vals))

    def store(self, *args):
        self.memory.push(*args)

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        q_pred = self.online(states)  # shape: (B, A)
        q_pred = q_pred[np.arange(BATCH_SIZE), actions]

        q_next = self.target(next_states)
        q_max = np.max(q_next, axis=1)
        q_target = rewards + GAMMA * q_max * (1.0 - dones)

        diff = (q_pred - q_target)
        loss_grad = diff / BATCH_SIZE

        grad_out = np.zeros((BATCH_SIZE, self.action_dim), dtype=np.float32)
        grad_out[np.arange(BATCH_SIZE), actions] = loss_grad

        self.online.backward(grad_out)
        self.online.step(LR)

        self.eps = max(EPS_MIN, self.eps * EPS_DECAY)
        self.t_step += 1
        if self.t_step % TARGET_SYNC == 0:
            self.target.copy_from(self.online)

# ============================================================
# ------------------  TRAINING LOOP  -------------------------
# ============================================================
def vectorize_state(state_tuple):
    return np.array(state_tuple, dtype=np.float32)

def moving_average(x, window=100):
    """
    Calculate the simple moving average (MA) of the list x with the given window length.
    Returns a new array with length len(x) - window + 1.
    """
    weights = np.ones(window) / window
    return np.convolve(x, weights, mode='valid')


def train():
    env = SnakeGame(render=False)
    agent = DQNAgent(state_dim=11, action_dim=3)

    best_len = 0
    global_step = 0
    rewards = []  # List to store total reward of each episode for visualization with a plot

    for ep in range(1, EPISODES + 1):
        s = env.reset()
        sv = vectorize_state(s)
        total_r, steps = 0.0, 0

        while True:
            a = agent.choose_action(sv)
            s2, r, done = env.step(a)
            sv2 = vectorize_state(s2)

            agent.store(sv, a, r, sv2, float(done))
            agent.update()

            sv = sv2
            total_r += r
            steps += 1
            global_step += 1

            # # When setting render=True to visualize the training process,
            # only render during the final episodes to avoid greatly slowing down
            '''
            if ep>=1950:
                env.render()
            '''

            if done or steps >= MAX_STEPS:
                break

        rewards.append(total_r)

        if len(env.snake) > best_len:
            best_len = len(env.snake)
            np.savez(os.path.join(LOG_DIR, "best_DQN.npz"),
                     *[p.val for p in agent.online.params])

        logging.info(f"EP {ep:04d} | steps:{steps:03d} | reward:{total_r:7.2f} | length:{len(env.snake):02d} | eps:{agent.eps:.3f}")

        if ep % 100 == 0:
            print(f"[Ep {ep}] score={len(env.snake)}  eps={agent.eps:.3f}  best={best_len}")

    logging.info("==== Training finished ====")

    #---------- VISUALIZE (plotting the results) ----------
    # Choose the window size (e.g., 100 episodes)
    window = 100

    # Calculate the Moving Average (MA)
    smoothed = moving_average(rewards, window=window)

    # Prepare the x-axis for the MA
    #    since smoothed has length = len(rewards) - window + 1,
    #    shift the index so that smoothed[i] corresponds to episode = i + (window - 1)
    episodes_ma = range(window-1, len(rewards))

    # Plot both Raw and Smoothed curves
    plt.figure(figsize=(10,5))
    plt.plot(rewards, alpha=0.3, label='Raw Reward')             # raw reward curve
    plt.plot(episodes_ma, smoothed, label=f'MA (window={window})')  # smoothed curve
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Reward per Episode (with {window}-episode MA)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train()
 