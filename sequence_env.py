"""
sequence_env.py
===============
Full Sequence board game environment for RL.

Observation (per player, partial information):
  - Board planes    : 10×10 × 3  = 300  (own chips, opp chips, corners)
  - Own hand        : 104 binary         (which cards held, 2-deck indexed)
  - Discard counts  : 104 floats         (how many of each card played so far / 2)
  - Opp hand size   :   1 float          (normalised 0-1)
  Total             : 509

Action encoding: hand_idx * 100 + row * 10 + col  →  int in [0, 699]
"""

import numpy as np
import random
from copy import deepcopy
from typing import List, Tuple, Optional, Dict

# ── Card constants ─────────────────────────────────────────────────────────────
# Suits: 0=♠  1=♥  2=♦  3=♣     Ranks: 2..14  (J=11, Q=12, K=13, A=14)
SUITS = [0, 1, 2, 3]
RANKS = list(range(2, 15))   # 2..14

OBS_SIZE    = 509
ACTION_DIM  = 700   # 7 cards × 100 positions
HAND_SIZE   = 7     # 2-player game


def make_deck() -> List[Tuple[int,int]]:
    return [(s, r) for s in SUITS for r in RANKS]   # 52 cards


# ── Card → index in the 104-slot two-deck encoding ────────────────────────────
def card_to_idx(card: Tuple[int,int]) -> int:
    s, r = card
    return s * 13 + (r - 2)   # 0..51 within one deck half


# ── Official 10×10 board layout ────────────────────────────────────────────────
# Each cell is (suit, rank) or None for the four corners.
SP = 0   # Spades
HT = 1   # Hearts
DM = 2   # Diamonds
CL = 3   # Clubs
# Two copies of every non-Jack card appear across the board.
BOARD_LAYOUT = [
    [None, (DM,2), (DM,3), (DM,4), (DM,5), (DM,6), (DM,7), (DM,8), (DM,9), None],
    [(CL,10), (DM,2), (DM,3), (DM,4), (DM,5), (DM,6), (DM,7), (DM,8), (DM,9), (CL,13)],
    [(CL,12), (DM,10), (HT,2), (HT,3), (HT,4), (HT,5), (HT,6), (HT,7), (DM,14), (CL,14)],
    [(CL,11), (DM,12), (HT,8), (SP,2), (SP,3), (SP,4), (SP,5), (HT,8), (DM,13), (SP,2)],
    [(CL,10), (DM,11), (HT,9), (SP,6), (HT,2), (HT,3), (SP,6), (HT,9), (DM,12), (SP,3)],
    [(CL,9), (DM,10), (HT,10), (SP,7), (HT,4), (HT,5), (SP,7), (HT,10), (DM,11), (SP,4)],
    [(CL,8), (DM,9), (HT,12), (SP,8), (HT,6), (HT,7), (SP,8), (HT,12), (DM,10), (SP,5)],
    [(CL,7), (DM,8), (HT,13), (SP,9), (SP,14), (SP,13), (SP,9), (HT,13), (DM,9), (SP,6)],
    [(CL,6), (DM,7), (DM,6), (DM,5), (DM,4), (DM,3), (DM,2), (HT,14), (DM,8), (SP,7)],
    [None, (SP,9), (SP,8), (SP,7), (SP,6), (SP,5), (SP, 4), (SP,3), (SP,2), None],
]
BOARD_SIZE = 10
CORNERS    = {(0,0), (0,9), (9,0), (9,9)}
SEQ_LEN    = 5    # chips in a row to form a sequence
WIN_SEQS   = 2    # sequences needed to win (2-player)

# Precompute card → [(row, col), ...]
CARD_TO_POS: Dict[Tuple, List[Tuple[int,int]]] = {}
for _r in range(BOARD_SIZE):
    for _c in range(BOARD_SIZE):
        _card = BOARD_LAYOUT[_r][_c]
        if _card is not None:
            CARD_TO_POS.setdefault(_card, []).append((_r, _c))

# ── Jack helpers ───────────────────────────────────────────────────────────────
# One-eyed Jacks (♠J, ♥J): remove an opponent chip
# Two-eyed Jacks (♦J, ♣J): wild — place own chip anywhere empty

def is_one_eyed_jack(card):  return card[1] == 11 and card[0] in (0, 1)
def is_two_eyed_jack(card):  return card[1] == 11 and card[0] in (2, 3)
def is_jack(card):           return card[1] == 11


# ── Action encode / decode ─────────────────────────────────────────────────────
def encode_action(hand_idx: int, row: int, col: int) -> int:
    return hand_idx * 100 + row * 10 + col

def decode_action(a: int) -> Tuple[int, int, int]:
    hand_idx = a // 100
    rem      = a % 100
    return hand_idx, rem // 10, rem % 10


# ══════════════════════════════════════════════════════════════════════════════
class SequenceEnv:
    """
    Two-player Sequence environment.

    Board cell values
    -----------------
    0 = empty
    1 = player-0 chip
    2 = player-1 chip
    3 = corner (always counts as any player's chip)
    """

    def __init__(self):
        self.reset()

    # ── Reset ──────────────────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        for r, c in CORNERS:
            self.board[r, c] = 3

        # Two shuffled decks
        self.draw_pile    = make_deck() + make_deck()
        random.shuffle(self.draw_pile)
        self.discard_pile = []

        # Deal hands
        self.hands = [[], []]
        for _ in range(HAND_SIZE):
            for p in range(2):
                self.hands[p].append(self.draw_pile.pop())

        # Discard card-count tracker (2 × 52 = 104 slots)
        self.discard_counts = np.zeros(104, dtype=np.float32)

        self.current_player = 0
        self.done           = False
        self.winner         = None
        self.sequences      = [0, 0]
        self.turn_count     = 0

        return self.get_obs(0)

    # ── Observation ────────────────────────────────────────────────────────────
    def get_obs(self, player: int) -> np.ndarray:
        """
        Returns obs vector of length OBS_SIZE=509 from player's perspective.
        """
        own = player + 1
        opp = 2 - player          # 1→2, 0→1

        # Board planes (300)
        plane_own    = (self.board == own).astype(np.float32).flatten()
        plane_opp    = (self.board == opp).astype(np.float32).flatten()
        plane_corner = (self.board == 3 ).astype(np.float32).flatten()

        # Own hand (104 binary) — mark slots that are held
        hand_vec = np.zeros(104, dtype=np.float32)
        for card in self.hands[player]:
            idx = card_to_idx(card)
            # First copy (slots 0–51), then second copy (slots 52–103)
            if hand_vec[idx] == 0:
                hand_vec[idx] = 1.0
            else:
                hand_vec[idx + 52] = 1.0

        # Discard counts normalised (104)
        discard_vec = self.discard_counts / 2.0   # max 2 copies per card

        # Opponent hand size (1)
        opp_hand_size = np.array([len(self.hands[1 - player]) / HAND_SIZE],
                                 dtype=np.float32)

        return np.concatenate([
            plane_own, plane_opp, plane_corner,   # 300
            hand_vec,                              # 104
            discard_vec,                           # 104
            opp_hand_size,                         #   1
        ])

    # ── Legal actions ──────────────────────────────────────────────────────────
    def get_legal_actions(self, player: int) -> List[int]:
        """
        Returns encoded actions (int) that are currently legal for player.
        """
        actions = []
        own = player + 1
        opp = 2 - player
        hand = self.hands[player]

        for hi, card in enumerate(hand):
            if is_one_eyed_jack(card):
                # Remove any opponent chip NOT in a completed sequence
                for r in range(BOARD_SIZE):
                    for c in range(BOARD_SIZE):
                        if self.board[r, c] == opp and not self._in_locked_seq(r, c, opp):
                            actions.append(encode_action(hi, r, c))

            elif is_two_eyed_jack(card):
                # Place on any empty cell
                for r in range(BOARD_SIZE):
                    for c in range(BOARD_SIZE):
                        if self.board[r, c] == 0:
                            actions.append(encode_action(hi, r, c))

            else:
                # Normal card — must match a board position that's empty
                for r, c in CARD_TO_POS.get(card, []):
                    if self.board[r, c] == 0:
                        actions.append(encode_action(hi, r, c))

        return actions

    def _in_locked_seq(self, row: int, col: int, chip_val: int) -> bool:
        """True if (row,col) is part of a completed sequence for chip_val."""
        for seq in self._find_sequences(chip_val):
            if (row, col) in seq:
                return True
        return False

    # ── Step ───────────────────────────────────────────────────────────────────
    def step(self, action_enc: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Apply an encoded action for the current player.
        Returns (next_obs, reward, done, info).

        Rewards
        -------
        +1.0  win
        +0.5  complete a new sequence
        -0.5  opponent completes a sequence (shaped penalty)
        -0.01 per turn (speed incentive)
        """
        assert not self.done
        p   = self.current_player
        own = p + 1
        opp = 2 - p
        hi, row, col = decode_action(action_enc)

        card = self.hands[p][hi]

        # ── Apply action ───────────────────────────────────────────────────────
        if is_one_eyed_jack(card):
            self.board[row, col] = 0          # remove opponent chip
        else:
            self.board[row, col] = own         # place own chip

        # ── Discard & draw ─────────────────────────────────────────────────────
        played = self.hands[p].pop(hi)
        self.discard_pile.append(played)
        # Track discard
        idx = card_to_idx(played)
        if self.discard_counts[idx] < 1:
            self.discard_counts[idx] += 1
        else:
            self.discard_counts[idx + 52] += 1

        if not self.draw_pile:
            # Reshuffle discard (keep top card)
            self.draw_pile    = self.discard_pile[:-1]
            self.discard_pile = [self.discard_pile[-1]]
            random.shuffle(self.draw_pile)

        if self.draw_pile:
            self.hands[p].append(self.draw_pile.pop())

        # ── Score ──────────────────────────────────────────────────────────────
        seqs_before = self.sequences[:]
        self.sequences[0] = len(self._find_sequences(1))
        self.sequences[1] = len(self._find_sequences(2))

        reward = -0.01   # step penalty

        new_own_seqs = self.sequences[p]   - seqs_before[p]
        new_opp_seqs = self.sequences[1-p] - seqs_before[1-p]

        reward += 0.5 * new_own_seqs
        reward -= 0.5 * new_opp_seqs

        if self.sequences[p] >= WIN_SEQS:
            self.done   = True
            self.winner = p
            reward      = 1.0
        elif self.sequences[1-p] >= WIN_SEQS:
            self.done   = True
            self.winner = 1 - p
            reward      = -1.0

        self.turn_count    += 1
        self.current_player = 1 - p

        next_obs = self.get_obs(self.current_player)
        return next_obs, reward, self.done, {}

    # ── Handle dead cards (no legal position) ──────────────────────────────────
    def discard_dead(self, player: int) -> int:
        """
        Find and discard dead cards for player, draw replacements.
        Returns number of cards swapped.
        """
        swapped = 0
        i = 0
        while i < len(self.hands[player]):
            card = self.hands[player][i]
            if not is_jack(card) and all(
                self.board[r, c] != 0
                for r, c in CARD_TO_POS.get(card, [])
            ):
                dead = self.hands[player].pop(i)
                self.discard_pile.append(dead)
                if self.draw_pile:
                    self.hands[player].append(self.draw_pile.pop())
                swapped += 1
            else:
                i += 1
        return swapped

    # ── Sequence detection ─────────────────────────────────────────────────────
    def _find_sequences(self, chip_val: int):
        """Return list of completed sequences (each a frozenset of positions)."""
        found = []
        b = self.board

        def valid(r, c):
            return b[r, c] == chip_val or b[r, c] == 3

        for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    cells = []
                    for k in range(SEQ_LEN):
                        nr, nc = r + dr*k, c + dc*k
                        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and valid(nr, nc):
                            cells.append((nr, nc))
                        else:
                            break
                    if len(cells) == SEQ_LEN:
                        fs = frozenset(cells)
                        if fs not in found:
                            found.append(fs)
        return found

    # ── Render ─────────────────────────────────────────────────────────────────
    def render(self):
        sym = {0:'·', 1:'●', 2:'○', 3:'★'}
        print("\n    " + "  ".join(f"{c}" for c in range(BOARD_SIZE)))
        for r in range(BOARD_SIZE):
            row = f" {r:2} " + "  ".join(sym[self.board[r,c]] for c in range(BOARD_SIZE))
            print(row)
        print(f"\n  Seqs  P0:{self.sequences[0]}  P1:{self.sequences[1]}"
              f"   Turn:{self.turn_count}   Next:P{self.current_player}")