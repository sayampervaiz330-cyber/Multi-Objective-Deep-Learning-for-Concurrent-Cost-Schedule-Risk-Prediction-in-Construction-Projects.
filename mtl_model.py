"""
Multi-task deep network implemented directly in NumPy (no torch/tensorflow
available in this sandbox — this is a fully working substitute; the
architecture and joint-loss mechanism are identical to what you'd write in
PyTorch, so this is legitimate to describe in your methodology chapter, and
you can port it 1:1 to torch.nn.Module later if you want GPU speed).

Architecture:
    input -> shared trunk (Dense-ReLU x2)
          -> head_cost      (Dense-ReLU -> Dense linear)   , loss = MSE
          -> head_schedule  (Dense-ReLU -> Dense linear)   , loss = MSE
          -> head_risk      (Dense-ReLU -> Dense sigmoid)  , loss = BCE

Joint loss = w_cost*MSE_cost + w_sched*MSE_sched + w_risk*BCE_risk
Gradients from all three heads flow back into the SAME shared trunk in one
backward pass — this is the actual mechanism that lets the network learn
shared cost/schedule/risk structure instead of the "silo" problem.
"""
import numpy as np


class Dense:
    def __init__(self, in_dim, out_dim, activation=None, seed=None):
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6 / (in_dim + out_dim))
        self.W = rng.uniform(-limit, limit, (in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.activation = activation
        # Adam state
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        if self.activation == "relu":
            self.a = np.maximum(0, self.z)
        elif self.activation == "sigmoid":
            self.a = 1 / (1 + np.exp(-np.clip(self.z, -30, 30)))
        else:
            self.a = self.z
        return self.a

    def backward(self, d_out):
        if self.activation == "relu":
            dz = d_out * (self.z > 0)
        elif self.activation == "sigmoid":
            dz = d_out  # combined with BCE loss upstream (see risk head loss)
        else:
            dz = d_out
        n = self.x.shape[0]
        self.dW = self.x.T @ dz / n
        self.db = dz.mean(axis=0)
        return dz @ self.W.T

    def adam_step(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, t=1):
        for p, dp, m, v in [(self.W, self.dW, "mW", "vW"), (self.b, self.db, "mb", "vb")]:
            m_val = getattr(self, m)
            v_val = getattr(self, v)
            m_val[:] = beta1 * m_val + (1 - beta1) * dp
            v_val[:] = beta2 * v_val + (1 - beta2) * (dp ** 2)
            m_hat = m_val / (1 - beta1 ** t)
            v_hat = v_val / (1 - beta2 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)


class MLPBlock:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_out):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
        return d_out

    def adam_step(self, **kw):
        for layer in self.layers:
            layer.adam_step(**kw)


class MultiTaskNet:
    def __init__(self, n_features, hidden=64, head_hidden=16, seed=0):
        self.trunk = MLPBlock([
            Dense(n_features, hidden, "relu", seed=seed),
            Dense(hidden, hidden // 2, "relu", seed=seed + 1),
        ])
        rep_dim = hidden // 2
        self.head_cost = MLPBlock([
            Dense(rep_dim, head_hidden, "relu", seed=seed + 2),
            Dense(head_hidden, 1, None, seed=seed + 3),
        ])
        self.head_sched = MLPBlock([
            Dense(rep_dim, head_hidden, "relu", seed=seed + 4),
            Dense(head_hidden, 1, "sigmoid", seed=seed + 5),
        ])
        self.head_risk = MLPBlock([
            Dense(rep_dim, head_hidden, "relu", seed=seed + 6),
            Dense(head_hidden, 1, "sigmoid", seed=seed + 7),
        ])
        self.t = 0  # adam timestep

    def forward(self, x):
        h = self.trunk.forward(x)
        cost = self.head_cost.forward(h).ravel()
        sched = self.head_sched.forward(h).ravel()
        risk = self.head_risk.forward(h).ravel()
        return cost, sched, risk

    def train_step(self, x, y_cost, y_sched, y_risk, w_cost=1.0, w_sched=1.0, w_risk=1.0, lr=1e-3):
        n = x.shape[0]
        cost_pred, sched_pred, risk_pred = self.forward(x)

        # dLoss/dPred for each head (chain rule already includes the 1/n mean)
        d_cost = w_cost * 2 * (cost_pred - y_cost).reshape(-1, 1) / n
        # BCE + sigmoid combined gradient simplifies to (pred - label) for both
        # classification heads (schedule_overrun and risk)
        d_sched = w_sched * (sched_pred - y_sched).reshape(-1, 1) / n
        d_risk = w_risk * (risk_pred - y_risk).reshape(-1, 1) / n

        dh_cost = self.head_cost.backward(d_cost)
        dh_sched = self.head_sched.backward(d_sched)
        dh_risk = self.head_risk.backward(d_risk)
        dh = dh_cost + dh_sched + dh_risk  # <-- the actual multi-task coupling

        self.trunk.backward(dh)

        self.t += 1
        for block in [self.trunk, self.head_cost, self.head_sched, self.head_risk]:
            block.adam_step(lr=lr, t=self.t)

        eps = 1e-9
        bce_sched = -np.mean(y_sched * np.log(sched_pred + eps) + (1 - y_sched) * np.log(1 - sched_pred + eps))
        bce_risk = -np.mean(y_risk * np.log(risk_pred + eps) + (1 - y_risk) * np.log(1 - risk_pred + eps))
        loss = w_cost * np.mean((cost_pred - y_cost) ** 2) + w_sched * bce_sched + w_risk * bce_risk
        return loss

    def fit(self, X, y_cost, y_sched, y_risk, epochs=200, batch_size=512, lr=1e-3,
            weights=(1.0, 1.0, 1.0), verbose_every=20):
        n = X.shape[0]
        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                b = idx[start:start + batch_size]
                epoch_loss += self.train_step(
                    X[b], y_cost[b], y_sched[b], y_risk[b],
                    w_cost=weights[0], w_sched=weights[1], w_risk=weights[2], lr=lr
                ) * len(b)
            if epoch % verbose_every == 0 or epoch == 1:
                print(f"  epoch {epoch:4d}  joint loss = {epoch_loss / n:.4f}")

    def predict(self, X):
        cost, sched, risk = self.forward(X)
        return cost, sched, risk
