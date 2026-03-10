"""
train.py
========
Self-play training loop for NFSP Sequence agents.

Usage
-----
  python train.py                      # default 3000 episodes
  python train.py --episodes 10000
  python train.py --episodes 500 --eval_every 100

What happens each episode
--------------------------
1. Two agents play a full game, alternating turns.
2. Each agent uses either its BR or AVG policy (mixed by η).
3. BR transitions accumulate in each agent's RL buffer.
4. Every `update_every` steps, PPO updates BRNet and SL updates AVGNet.
5. Every `eval_every` episodes, both agents play greedily (η=0) for N games
   and we record win rates.
6. Plots and checkpoints are saved periodically.
"""

import os, time, argparse, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sequence_env import SequenceEnv, ACTION_DIM
from nfsp_agent   import NFSPAgent

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    episodes      = 3000,
    max_turns     = 400,
    update_every  = 256,    # PPO update frequency (transitions per agent)
    eval_every    = 200,
    eval_episodes = 50,
    save_every    = 1000,
    device        = "cuda" if torch.cuda.is_available() else "cpu",
    out_dir       = "output",
)


# ══════════════════════════════════════════════════════════════════════════════
def make_legal_mask_tensor(legal_actions, device):
    mask = torch.zeros(ACTION_DIM, dtype=torch.bool, device=device)
    for a in legal_actions:
        if 0 <= a < ACTION_DIM:
            mask[a] = True
    return mask


def play_episode(env: SequenceEnv, agents: list,
                 max_turns: int = 400, training: bool = True) -> dict:
    """
    Run one game.  Agents learn from transitions during the episode.
    Returns result dict.
    """
    obs    = env.reset()
    prev   = [{}, {}]   # metadata from previous step for each agent
    wins   = [0, 0]
    total_reward = [0.0, 0.0]

    for _ in range(max_turns):
        if env.done:
            break

        p     = env.current_player
        agent = agents[p]

        # Discard dead cards first
        env.discard_dead(p)

        legal = env.get_legal_actions(p)
        if not legal:
            env.current_player = 1 - p   # forced pass
            continue

        action, meta = agent.select_action(obs, legal, training=training)
        if action is None:
            env.current_player = 1 - p
            continue

        next_obs, reward, done, _ = env.step(action)

        # Opponent gets negative of the reward as signal
        opp_reward = -reward

        # Store this player's transition
        if training and meta.get('used_br', False):
            agent.store_transition(
                obs, action, meta['log_prob'], meta['value'],
                reward, float(done), meta['mask']
            )

        # If the previous step for the *other* player is pending, close it
        opp = 1 - p
        if training and prev[opp]:
            pm = prev[opp]
            if pm.get('used_br', False):
                agents[opp].store_transition(
                    pm['obs'], pm['action'], pm['log_prob'], pm['value'],
                    opp_reward, float(done), pm['mask']
                )

        prev[p] = dict(obs=obs, action=action, **meta)
        total_reward[p] += reward
        obs = next_obs

    # PPO + SL updates
    br_loss  = [None, None]
    avg_loss = [None, None]
    if training:
        for p, agent in enumerate(agents):
            if agent.rl_buffer.is_full() or env.done:
                br_loss[p]  = agent.update_br()
                avg_loss[p] = agent.update_avg()

    return {
        'winner':   env.winner,
        'turns':    env.turn_count,
        'rewards':  total_reward,
        'br_loss':  br_loss,
        'avg_loss': avg_loss,
        'seqs':     env.sequences[:],
    }


def evaluate(env: SequenceEnv, agents: list, n: int = 50) -> dict:
    """Run n greedy games (η=0 for both agents)."""
    saved_eta = [a.eta for a in agents]
    for a in agents:
        a.eta = 0.0   # use AVG policy only

    wins      = [0, 0, 0]   # p0, p1, draw/timeout
    turn_list = []
    for _ in range(n):
        r = play_episode(env, agents, max_turns=400, training=False)
        if r['winner'] == 0:   wins[0] += 1
        elif r['winner'] == 1: wins[1] += 1
        else:                  wins[2] += 1
        turn_list.append(r['turns'])

    for i, a in enumerate(agents):
        a.eta = saved_eta[i]

    return {
        'p0_wr':    wins[0] / n,
        'p1_wr':    wins[1] / n,
        'draw_rate':wins[2] / n,
        'avg_turns':float(np.mean(turn_list)),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────
def save_plots(history: dict, path: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor('#111122')
    fig.suptitle('NFSP Sequence — Self-Play Training', color='white', fontsize=14)

    evals = history.get('evals', [])
    ep_x  = [e['episode'] for e in evals]

    # ── Win rate ──────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a2e')
    if evals:
        ax.plot(ep_x, [e['p0_wr']    for e in evals], color='#00d4ff', lw=2, label='P0 win%')
        ax.plot(ep_x, [e['p1_wr']    for e in evals], color='#ff6b6b', lw=2, label='P1 win%')
        ax.plot(ep_x, [e['draw_rate'] for e in evals], color='#ffd700', lw=1.5,
                ls='--', label='Draw%')
        ax.axhline(0.5, color='#555', ls=':', lw=1)
        ax.set_ylim(0, 1)
        ax.legend(facecolor='#222', labelcolor='white', fontsize=9)
    ax.set_title('Win Rate (greedy eval)', color='white')
    ax.set_xlabel('Episode', color='white')
    ax.tick_params(colors='white')

    # ── Average game length ───────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a2e')
    if evals:
        ax.plot(ep_x, [e['avg_turns'] for e in evals], color='#a8ff78', lw=2)
    ax.set_title('Avg Game Length', color='white')
    ax.set_xlabel('Episode', color='white')
    ax.set_ylabel('Turns', color='white')
    ax.tick_params(colors='white')

    # ── BR loss ───────────────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a2e')
    br_losses = history.get('br_losses', [])
    if br_losses:
        smooth = np.convolve(br_losses, np.ones(20)/20, mode='valid')
        ax.plot(br_losses, color='#ff6b6b', alpha=0.25, lw=0.8)
        ax.plot(smooth,    color='#ff6b6b', lw=2, label='BR loss (smoothed)')
        ax.legend(facecolor='#222', labelcolor='white', fontsize=9)
    ax.set_title('BR Network Loss (PPO)', color='white')
    ax.set_xlabel('Update step', color='white')
    ax.tick_params(colors='white')

    # ── AVG loss ──────────────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a2e')
    avg_losses = history.get('avg_losses', [])
    if avg_losses:
        smooth = np.convolve(avg_losses, np.ones(20)/20, mode='valid')
        ax.plot(avg_losses, color='#a78bfa', alpha=0.25, lw=0.8)
        ax.plot(smooth,     color='#a78bfa', lw=2, label='AVG loss (smoothed)')
        ax.legend(facecolor='#222', labelcolor='white', fontsize=9)
    ax.set_title('AVG Network Loss (Supervised)', color='white')
    ax.set_xlabel('Update step', color='white')
    ax.tick_params(colors='white')

    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Plots saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
def train(cfg: dict):
    os.makedirs(cfg['out_dir'], exist_ok=True)
    device = cfg['device']
    print(f"\n{'='*60}")
    print(f"  NFSP Sequence Self-Play Training")
    print(f"  Episodes : {cfg['episodes']}  |  Device : {device}")
    print(f"{'='*60}\n")

    env    = SequenceEnv()
    agents = [
        NFSPAgent(player_id=0, device=device),
        NFSPAgent(player_id=1, device=device),
    ]

    history = dict(evals=[], br_losses=[], avg_losses=[])

    # Running stats
    window  = 100
    win_buf = [[], []]
    t_start = time.time()

    for ep in range(1, cfg['episodes'] + 1):

        result = play_episode(env, agents, max_turns=cfg['max_turns'], training=True)

        # Track wins
        for p in range(2):
            win_buf[p].append(1 if result['winner'] == p else 0)
            if len(win_buf[p]) > window:
                win_buf[p].pop(0)

        # Collect losses
        for p in range(2):
            if result['br_loss'][p] is not None:
                history['br_losses'].append(result['br_loss'][p])
            if result['avg_loss'][p] is not None:
                history['avg_losses'].append(result['avg_loss'][p])

        # ── Console log every 50 episodes ─────────────────────────────────────
        if ep % 50 == 0:
            elapsed = time.time() - t_start
            p0_wr   = np.mean(win_buf[0]) if win_buf[0] else 0
            p1_wr   = np.mean(win_buf[1]) if win_buf[1] else 0
            eta0    = agents[0].eta
            eta1    = agents[1].eta
            br_l    = f"{history['br_losses'][-1]:.4f}"  if history['br_losses']  else "—"
            avg_l   = f"{history['avg_losses'][-1]:.4f}" if history['avg_losses'] else "—"
            print(
                f"  Ep {ep:5d}/{cfg['episodes']}"
                f"  |  P0 wr={p0_wr:.2f}  P1 wr={p1_wr:.2f}"
                f"  |  η=({eta0:.3f},{eta1:.3f})"
                f"  |  BR loss={br_l}  AVG loss={avg_l}"
                f"  |  turns={result['turns']:3d}"
                f"  |  {elapsed:.0f}s"
            )

        # ── Evaluation ────────────────────────────────────────────────────────
        if ep % cfg['eval_every'] == 0:
            print(f"\n  --- Evaluating at episode {ep} ---")
            ev = evaluate(env, agents, n=cfg['eval_episodes'])
            ev['episode'] = ep
            history['evals'].append(ev)
            print(
                f"  P0 win rate : {ev['p0_wr']:.3f}"
                f"  P1 win rate : {ev['p1_wr']:.3f}"
                f"  draws : {ev['draw_rate']:.3f}"
                f"  avg turns : {ev['avg_turns']:.1f}\n"
            )

        # ── Save checkpoints ──────────────────────────────────────────────────
        if ep % cfg['save_every'] == 0:
            for p, agent in enumerate(agents):
                agent.save(os.path.join(cfg['out_dir'], f"agent_p{p}_ep{ep}.pt"))

        # ── Plots ─────────────────────────────────────────────────────────────
        if ep % cfg['eval_every'] == 0:
            save_plots(history, os.path.join(cfg['out_dir'], 'training_curves.png'))

    # ── Final save ─────────────────────────────────────────────────────────────
    for p, agent in enumerate(agents):
        agent.save(os.path.join(cfg['out_dir'], f"agent_p{p}_final.pt"))

    save_plots(history, os.path.join(cfg['out_dir'], 'training_curves.png'))

    # Save history
    with open(os.path.join(cfg['out_dir'], 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n  Training complete in {(time.time()-t_start)/60:.1f} min")
    print(f"  Outputs in → {cfg['out_dir']}/")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NFSP Sequence Self-Play')
    parser.add_argument('--episodes',      type=int, default=DEFAULTS['episodes'])
    parser.add_argument('--max_turns',     type=int, default=DEFAULTS['max_turns'])
    parser.add_argument('--eval_every',    type=int, default=DEFAULTS['eval_every'])
    parser.add_argument('--eval_episodes', type=int, default=DEFAULTS['eval_episodes'])
    parser.add_argument('--save_every',    type=int, default=DEFAULTS['save_every'])
    parser.add_argument('--out_dir',       type=str, default=DEFAULTS['out_dir'])
    parser.add_argument('--device',        type=str, default=DEFAULTS['device'])
    args = parser.parse_args()

    cfg = DEFAULTS.copy()
    cfg.update(vars(args))
    train(cfg)