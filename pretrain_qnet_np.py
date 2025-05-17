import os
import numpy as np

# —————————— Copy class Neural from snake_dqn.py ——————————
class Node:
    def __init__(self, val):
        self.val  = val.astype(np.float32, copy=False)
        self.grad = np.zeros_like(self.val, dtype=np.float32)
    def zero_grad(self):
        self.grad.fill(0.)

class Layer:
    def __init__(self, in_dim, out_dim, act_relu=True):
        limit = 1. / np.sqrt(in_dim)
        self.W = Node(np.random.uniform(-limit, limit, (in_dim, out_dim)))
        self.b = Node(np.zeros(out_dim, dtype=np.float32))
        self.act_relu = act_relu
        self._last_x = None
        self._mask   = None

    def forward(self, x):
        self._last_x = x
        y = x @ self.W.val + self.b.val
        if self.act_relu:
            self._mask = (y > 0).astype(np.float32)
            y = y * self._mask
        return y

    def backward(self, grad_out):
        if self.act_relu:
            grad_out = grad_out * self._mask
        x = self._last_x
        self.W.grad += x.T @ grad_out
        self.b.grad += grad_out.sum(axis=0)
        return grad_out @ self.W.val.T

class Neural:
    def __init__(self, input_dim, output_dim):
        self.layers = [
            Layer(input_dim, 64, act_relu=True),
            Layer(64,       64, act_relu=True),
            Layer(64, output_dim, act_relu=False),
        ]
        self.params = []
        for lyr in self.layers:
            self.params += [lyr.W, lyr.b]

    def __call__(self, x):
        for lyr in self.layers:
            x = lyr.forward(x)
        return x

    def backward(self, grad_out):
        for lyr in reversed(self.layers):
            grad_out = lyr.backward(grad_out)

    def step(self, lr):
        for p in self.params:
            p.val -= lr * p.grad
            p.zero_grad()
# —————————— end copy ——————————

def int_to_bits(idx, bits=11):
    return np.array([(idx >> i) & 1 for i in reversed(range(bits))],
                    dtype=np.float32)

def main():
    # Configuration
    LOG_DIR = "logs"
    PRETRAIN_EPOCHS = 2000
    BATCH_SIZE      = 128
    LR              = 1e-3

    qtable = np.load(os.path.join(LOG_DIR, "qtable_final.npy")).astype(np.float32)

    # Build states array (2048×11)
    N = 2**11
    states = np.stack([int_to_bits(i) for i in range(N)], axis=0)

    # Initialize network and pre-train
    net = Neural(input_dim=11, output_dim=3)

    for ep in range(1, PRETRAIN_EPOCHS+1):
        perm = np.random.permutation(N)
        total_loss = 0.0

        for i in range(0, N, BATCH_SIZE):
            idxs = perm[i : i + BATCH_SIZE]
            x = states[idxs]      # (B,11)
            y = qtable[idxs]      # (B,3)

            pred = net(x)         # forward
            diff = pred - y
            loss = np.mean(diff**2)
            total_loss += loss * x.shape[0]

            # grad_out = dL/dpred = 2*(pred - y)/B
            grad_out = (2 * diff) / x.shape[0]
            net.backward(grad_out)
            net.step(LR)

        if ep % 50 == 0:
            print(f"[Pretrain] Epoch {ep}/{PRETRAIN_EPOCHS}, loss = {total_loss/N:.6f}")

    # Save pre-trained weights
    params = [p.val for p in net.params]
    np.savez("pretrained_qnet.npz", *params)
    print("Pre-training xong → pretrained_qnet.npz")

if __name__ == "__main__":
    main()
