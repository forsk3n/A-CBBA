"""
A-CBBA: Adaptive Consensus Stopping + Weighted Relay Selection
Modification of CBBA for imperfect communication channels.

Scientific problem addressed:
  Baseline CBBA uses a fixed number of consensus rounds determined by
  the diameter of a fully connected graph. Under lossy channels this
  causes premature termination (convergence failure) and under ideal
  channels causes redundant rounds (wasted bandwidth).

Solution:
  1. Adaptive stopping: each agent monitors local bid-table stability
     over τ consecutive rounds; global stop when all reachable agents
     are stable.
  2. Weighted relay selection: relay agent scored by
       score(k,d) = α·R_k + (1-α)·Q_{k,d}
     where R_k = reliability (battery proxy), Q_{k,d} = channel quality.
"""

import numpy as np
import math
from cbba_core import CBBA, Agent, Task, WorldInfo

# ─────────────────────────────────────────────────────────────────
#  Helper: apply Bernoulli packet loss to an adjacency matrix
# ─────────────────────────────────────────────────────────────────
def apply_packet_loss(adj_matrix, p_loss, rng=None):
    """Return a noisy copy of adj_matrix: each edge dropped with prob p_loss."""
    if rng is None:
        rng = np.random.default_rng()
    adj = np.array(adj_matrix, dtype=int).copy()
    n = adj.shape[0]
    np.fill_diagonal(adj, 0)
    mask = rng.random((n, n)) > p_loss          # True = link survives
    mask = np.triu(mask, 1)
    mask = mask | mask.T                         # keep symmetric
    adj = (adj & mask).astype(int)
    return adj.tolist()


# ─────────────────────────────────────────────────────────────────
#  A-CBBA: adaptive-stop CBBA subclass
# ─────────────────────────────────────────────────────────────────
class ACBBA(CBBA):
    """
    Extends CBBA with:
      - packet-loss simulation per consensus round
      - adaptive stopping based on bid-table stability (window τ)
      - weighted relay-agent selection (reliability × channel quality)
    """

    def __init__(self, agent_types, task_types, compatibility_mat,
                 p_loss=0.0, tau=3, alpha_relay=0.5, max_extra_rounds=30):
        super().__init__(agent_types, task_types, compatibility_mat)
        self.p_loss        = p_loss          # per-edge packet-loss probability
        self.tau           = tau             # stability window (rounds)
        self.alpha_relay   = alpha_relay     # weight for reliability vs channel quality
        self.max_extra_rounds = max_extra_rounds
        self._rng = np.random.default_rng(seed=42)

    # ── reliability coefficient (proxy: normalised fuel level) ────
    def _reliability(self, agent_idx):
        fuel = self.AgentList[agent_idx].fuel
        max_fuel = max((a.fuel for a in self.AgentList), default=1) or 1
        return float(fuel) / max_fuel

    # ── channel quality between two agents (distance-based) ───────
    def _channel_quality(self, src_idx, dst_idx):
        a = self.AgentList[src_idx]
        b = self.AgentList[dst_idx]
        dist = math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)
        max_range = 2000.0          # normalisation constant (arena diameter)
        return max(0.0, 1.0 - dist / max_range)

    # ── relay score ───────────────────────────────────────────────
    def _relay_score(self, relay_idx, dst_idx):
        R = self._reliability(relay_idx)
        Q = self._channel_quality(relay_idx, dst_idx)
        return self.alpha_relay * R + (1.0 - self.alpha_relay) * Q

    # ── build per-round effective graph with loss + relay routing ──
    def _effective_graph(self, base_graph):
        """
        1. Drop edges according to p_loss (Bernoulli).
        2. For each disconnected pair (i,j), find best relay k and
           add a multi-hop virtual edge if score(k,j) is above threshold.
        """
        n = self.num_agents
        lossy = apply_packet_loss(base_graph, self.p_loss, self._rng)
        lossy_np = np.array(lossy, dtype=int)

        # Try relay routing for pairs not directly connected
        effective = lossy_np.copy()
        for i in range(n):
            for j in range(n):
                if i == j or effective[i][j] == 1:
                    continue
                # find best relay among neighbours of i that can reach j
                best_score = -1.0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if lossy_np[i][k] == 1 and lossy_np[k][j] == 1:
                        s = self._relay_score(k, j)
                        if s > best_score:
                            best_score = s
                if best_score > 0.3:       # relay quality threshold
                    effective[i][j] = 1
                    effective[j][i] = 1

        np.fill_diagonal(effective, 0)
        return effective.tolist()

    # ── adaptive convergence check ─────────────────────────────────
    def _check_stability(self, history):
        """
        Returns True if every agent's winner_bid_list has been identical
        for the last τ rounds.
        history: list of snapshots of winner_bid_list (newest last)
        """
        if len(history) < self.tau:
            return False
        recent = history[-self.tau:]
        for agent in range(self.num_agents):
            for r in range(1, self.tau):
                if recent[r][agent] != recent[r-1][agent]:
                    return False
        return True

    # ── override solve() ──────────────────────────────────────────
    def solve(self, AgentList, TaskList, WorldInfoInput, max_depth,
              time_window_flag, graph=None):
        self.settings(AgentList, TaskList, WorldInfoInput,
                      max_depth, time_window_flag)
        base_graph = graph if graph is not None else self.graph
        base_graph_np = np.array(base_graph)

        iter_idx = 1
        time_mat = [[0]*self.num_agents for _ in range(self.num_agents)]
        bid_history = []        # for adaptive stopping
        done_flag  = False
        max_iter   = 500

        while not done_flag:
            if iter_idx > max_iter:
                break

            # build effective graph this round
            eff_graph = self._effective_graph(base_graph_np.tolist())
            self.graph = eff_graph

            time_mat = self.communicate(time_mat, iter_idx)

            for idx_agent in range(self.num_agents):
                self.bundle(idx_agent)

            # snapshot for stability check (deep copy per-agent rows)
            snapshot = [list(self.winner_bid_list[a]) for a in range(self.num_agents)]
            bid_history.append(snapshot)

            if self._check_stability(bid_history):
                done_flag = True
            else:
                iter_idx += 1

        self.consensus_rounds = iter_idx   # expose for metrics

        # post-processing identical to base class
        for n in range(self.num_agents):
            for m in range(self.max_depth):
                if self.bundle_list[n][m] == -1:
                    break
                else:
                    self.bundle_list[n][m] = self.TaskList[self.bundle_list[n][m]].task_id
                if self.path_list[n][m] == -1:
                    break
                else:
                    self.path_list[n][m] = self.TaskList[self.path_list[n][m]].task_id

        self.path_list  = [list(filter(lambda a: a != -1, self.path_list[i]))  for i in range(len(self.path_list))]
        self.times_list = [list(filter(lambda a: a != -1, self.times_list[i])) for i in range(len(self.times_list))]
        return self.path_list, self.times_list
