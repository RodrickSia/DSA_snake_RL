import os
import numpy as np
from snake_dqn import Neural

def int_to_bits(idx, bits=11):
    return np.array([(idx >> i) & 1 for i in reversed(range(bits))],
                    dtype=np.float32)

def train():
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
    print("Pre-training done → pretrained_qnet.npz")

if __name__ == "__main__":
    train()
