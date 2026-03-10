"""
nfsp_agent.py
=============
Neural Fictitious Self-Play (NFSP) agent.

Each agent holds:
  - BRNet  : best-response actor-critic (trained with PPO)
  - AVGNet : average strategy           (trained with supervised learning)
  - RL buffer  : rollout data for PPO updates
  - SL buffer  : (obs, action) pairs for AVG supervised updates

At each step the agent decides which policy to follow:
  - with probability η   → follow BR  (and log the action to SL buffer)
  - with probability 1-η → follow AVG

η is annealed from 0.5 → 0.1 over training so the agent gradually
shifts from exploration to exploitation.

PPO Hyperparameters (tuned for board games):
  clip_eps=0.2, epochs=4, mini_batches=4, entropy_coef=0.01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import List, Optional, Tuple

from sequence_env import OBS_SIZE, ACTION_DIM
from networks     import BRNet, AVGNet


# ── Replay buffers ─────────────────────────────────────────────────────────────
class RLBuffer:
    """
    On-policy rollout buffer for PPO.
    Stores a fixed number of transitions then flushes after each update.
    """
    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self.clear()

    def clear(self):
        self.obs       = []
        self.actions   = []
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.dones     = []
        self.masks     = []   # legal action masks stored for reuse

    def push(self, obs, action, log_prob, value, reward, done, mask):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.masks.append(mask)

    def __len__(self):
        return len(self.obs)

    def is_full(self):
        return len(self) >= self.capacity

    def to_tensors(self, device):
        obs       = torch.FloatTensor(np.array(self.obs)).to(device)
        actions   = torch.LongTensor(self.actions).to(device)
        log_probs = torch.FloatTensor(self.log_probs).to(device)
        values    = torch.FloatTensor(self.values).to(device)
        rewards   = torch.FloatTensor(self.rewards).to(device)
        dones     = torch.FloatTensor(self.dones).to(device)
        masks     = torch.BoolTensor(np.array(self.masks)).to(device)
        return obs, actions, log_probs, values, rewards, dones, masks


class SLBuffer:
    """
    Fixed-size circular buffer for supervised learning of AVGNet.
    Stores (obs, action) pairs from BR's greedy decisions.
    """
    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.obs      = deque(maxlen=capacity)
        self.actions  = deque(maxlen=capacity)

    def push(self, obs: np.ndarray, action: int):
        self.obs.append(obs)
        self.actions.append(action)

    def sample(self, batch_size: int, device):
        idx = random.sample(range(len(self)), batch_size)
        obs = torch.FloatTensor(np.array([self.obs[i]     for i in idx])).to(device)
        act = torch.LongTensor(         [self.actions[i]  for i in idx]).to(device)
        return obs, act

    def __len__(self):
        return len(self.obs)


# ══════════════════════════════════════════════════════════════════════════════
class NFSPAgent:
    """
    Single NFSP agent.  Two agents are created for self-play, sharing
    no weights — they learn independently.

    Parameters
    ----------
    player_id   : 0 or 1
    eta_start   : initial probability of following BR policy
    eta_end     : final probability after annealing
    eta_anneal  : number of steps over which η decays
    """

    def __init__(
        self,
        player_id:    int,
        eta_start:    float = 0.5,
        eta_end:      float = 0.1,
        eta_anneal:   int   = 50_000,
        br_lr:        float = 3e-4,
        avg_lr:       float = 1e-3,
        gamma:        float = 0.99,
        gae_lambda:   float = 0.95,
        clip_eps:     float = 0.2,
        ppo_epochs:   int   = 4,
        ppo_minibatch:int   = 512,
        entropy_coef: float = 0.01,
        vf_coef:      float = 0.5,
        sl_batch:     int   = 256,
        rl_capacity:  int   = 2048,
        sl_capacity:  int   = 200_000,
        device:       str   = "cpu",
    ):
        self.player_id     = player_id
        self.eta           = eta_start
        self.eta_end       = eta_end
        self.eta_decay     = (eta_start - eta_end) / eta_anneal
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_eps      = clip_eps
        self.ppo_epochs    = ppo_epochs
        self.ppo_minibatch = ppo_minibatch
        self.entropy_coef  = entropy_coef
        self.vf_coef       = vf_coef
        self.sl_batch      = sl_batch
        self.device        = torch.device(device)
        self.steps         = 0

        # Networks
        self.br_net  = BRNet().to(self.device)
        self.avg_net = AVGNet().to(self.device)

        # Optimizers
        self.br_optim  = torch.optim.Adam(self.br_net.parameters(),  lr=br_lr,  eps=1e-5)
        self.avg_optim = torch.optim.Adam(self.avg_net.parameters(), lr=avg_lr, eps=1e-5)

        # Buffers
        self.rl_buffer = RLBuffer(rl_capacity)
        self.sl_buffer = SLBuffer(sl_capacity)

        # Tracking
        self.using_br    = True   # which policy this agent used this turn
        self.br_losses   = []
        self.avg_losses  = []

    # ── Build legal mask ───────────────────────────────────────────────────────
    def _make_mask(self, legal_actions: List[int]) -> np.ndarray:
        mask = np.zeros(ACTION_DIM, dtype=bool)
        for a in legal_actions:
            if 0 <= a < ACTION_DIM:
                mask[a] = True
        return mask

    # ── Select action ──────────────────────────────────────────────────────────
    def select_action(self, obs: np.ndarray, legal_actions: List[int],
                      training: bool = True) -> Tuple[int, dict]:
        """
        Choose an action and return it alongside metadata needed for learning.

        Returns
        -------
        action  : encoded int
        meta    : dict with log_prob, value, mask, used_br
        """
        if not legal_actions:
            return None, {}

        mask_np = self._make_mask(legal_actions)
        mask_t  = torch.BoolTensor(mask_np).unsqueeze(0).to(self.device)
        obs_t   = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Decide which policy to use
        self.using_br = (random.random() < self.eta) if training else False

        if self.using_br:
            # BR policy
            with torch.no_grad():
                action_t, log_prob_t, _, value_t = self.br_net.get_action_and_value(
                    obs_t, mask_t
                )
            action   = action_t.item()
            log_prob = log_prob_t.item()
            value    = value_t.item()

            # Log to SL buffer so AVGNet can learn from this decision
            if training:
                self.sl_buffer.push(obs, action)

        else:
            # AVG policy
            with torch.no_grad():
                action_t = self.avg_net.sample(obs_t, mask_t)
            action   = action_t.item()
            log_prob = 0.0    # not used for AVG in PPO update
            value    = 0.0

        if action not in legal_actions:
            # Fallback safety (shouldn't happen with correct masking)
            action = random.choice(legal_actions)

        meta = dict(log_prob=log_prob, value=value, mask=mask_np, used_br=self.using_br)
        return action, meta

    # ── Store transition ───────────────────────────────────────────────────────
    def store_transition(self, obs, action, log_prob, value, reward, done, mask):
        """Only store in RL buffer when we were following BR policy."""
        if self.using_br:
            self.rl_buffer.push(obs, action, log_prob, value, reward, done, mask)

    # ── PPO update ─────────────────────────────────────────────────────────────
    def update_br(self) -> Optional[float]:
        """Run PPO on the RL buffer. Returns mean loss or None if buffer too small."""
        if len(self.rl_buffer) < 256:
            return None

        obs, actions, old_log_probs, old_values, rewards, dones, masks = \
            self.rl_buffer.to_tensors(self.device)

        # ── GAE advantage estimation ───────────────────────────────────────────
        with torch.no_grad():
            advantages = torch.zeros_like(rewards)
            last_gae   = 0.0
            for t in reversed(range(len(rewards))):
                is_last = (t == len(rewards) - 1)
                next_val = 0.0 if (dones[t] or is_last) else old_values[t+1].item()
                delta    = rewards[t] + self.gamma * next_val * (1 - dones[t]) - old_values[t]
                last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
                advantages[t] = last_gae

            returns = advantages + old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── PPO epochs ─────────────────────────────────────────────────────────
        n        = len(obs)
        indices  = np.arange(n)
        all_loss = []

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.ppo_minibatch):
                idx = indices[start:start + self.ppo_minibatch]
                if len(idx) < 2:
                    continue

                b_obs  = obs[idx];     b_act  = actions[idx]
                b_olp  = old_log_probs[idx]
                b_adv  = advantages[idx]
                b_ret  = returns[idx]
                b_mask = masks[idx]

                _, new_log_prob, entropy, new_value = \
                    self.br_net.get_action_and_value(b_obs, b_mask, b_act)

                ratio      = (new_log_prob - b_olp).exp()
                surr1      = ratio * b_adv
                surr2      = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = F.mse_loss(new_value, b_ret)
                ent_loss    = -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.entropy_coef * ent_loss

                self.br_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.br_net.parameters(), 0.5)
                self.br_optim.step()
                all_loss.append(loss.item())

        self.rl_buffer.clear()
        self.steps += 1
        self.eta = max(self.eta_end, self.eta - self.eta_decay)

        mean_loss = float(np.mean(all_loss)) if all_loss else 0.0
        self.br_losses.append(mean_loss)
        return mean_loss

    # ── Supervised update for AVGNet ───────────────────────────────────────────
    def update_avg(self) -> Optional[float]:
        """Train AVGNet via cross-entropy on SL buffer."""
        if len(self.sl_buffer) < self.sl_batch:
            return None

        obs_t, act_t = self.sl_buffer.sample(self.sl_batch, self.device)
        logits = self.avg_net(obs_t)
        loss   = F.cross_entropy(logits, act_t)

        self.avg_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.avg_net.parameters(), 0.5)
        self.avg_optim.step()

        l = loss.item()
        self.avg_losses.append(l)
        return l

    # ── Save / load ────────────────────────────────────────────────────────────
    def save(self, path: str):
        torch.save({
            'br_net':   self.br_net.state_dict(),
            'avg_net':  self.avg_net.state_dict(),
            'br_opt':   self.br_optim.state_dict(),
            'avg_opt':  self.avg_optim.state_dict(),
            'steps':    self.steps,
            'eta':      self.eta,
        }, path)
        print(f"  [P{self.player_id}] saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.br_net.load_state_dict(ckpt['br_net'])
        self.avg_net.load_state_dict(ckpt['avg_net'])
        self.br_optim.load_state_dict(ckpt['br_opt'])
        self.avg_optim.load_state_dict(ckpt['avg_opt'])
        self.steps = ckpt['steps']
        self.eta   = ckpt['eta']
        print(f"  [P{self.player_id}] loaded ← {path}")