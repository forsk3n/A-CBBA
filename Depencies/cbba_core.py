# cbba_core.py — Оригинальная версия, очищенная только от действительно неиспользуемого кода
import numpy as np
import math
import copy

# --- Структуры данных ---
class Agent:
    def __init__(self, agent_id, agent_type, x, y, z=0, nom_velocity=10, availability=0, fuel=0):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.x = x
        self.y = y
        self.z = z
        self.nom_velocity = nom_velocity
        self.availability = availability
        self.fuel = fuel

class Task:
    def __init__(self, task_id, task_type, x, y, z=0, start_time=0, end_time=1000, duration=1, task_value=10, discount=0.01):
        self.task_id = task_id
        self.task_type = task_type
        self.x = x
        self.y = y
        self.z = z
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.task_value = task_value
        self.discount = discount

class WorldInfo:
    def __init__(self, limit_x, limit_y, limit_z):
        self.limit_x = limit_x
        self.limit_y = limit_y
        self.limit_z = limit_z

# --- Реализация алгоритма CBBA ---
class CBBA(object):
    def __init__(self, agent_types, task_types, compatibility_mat):
        self.agent_types = agent_types
        self.task_types = task_types
        self.compatibility_mat = compatibility_mat
        self.comm_count = 0
        self.time_window_flag = True

    def settings(self, AgentList, TaskList, WorldInfoInput, max_depth, time_window_flag):
        self.num_agents = len(AgentList)
        self.num_tasks = len(TaskList)
        self.max_depth = max_depth
        self.time_window_flag = time_window_flag
        self.AgentList = AgentList
        self.TaskList = TaskList
        self.WorldInfo = WorldInfoInput
        self.space_limit_x = self.WorldInfo.limit_x
        self.space_limit_y = self.WorldInfo.limit_y
        self.space_limit_z = self.WorldInfo.limit_z
        self.graph = np.logical_not(np.identity(self.num_agents)).tolist()
        self.bundle_list = [[-1] * self.max_depth for _ in range(self.num_agents)]
        self.path_list = [[-1] * self.max_depth for _ in range(self.num_agents)]
        self.times_list = [[-1] * self.max_depth for _ in range(self.num_agents)]
        self.scores_list = [[-1] * self.max_depth for _ in range(self.num_agents)]
        self.bid_list = [[-1] * self.num_tasks for _ in range(self.num_agents)]
        self.winners_list = [[-1] * self.num_tasks for _ in range(self.num_agents)]
        self.winner_bid_list = [[-1] * self.num_tasks for _ in range(self.num_agents)]
        self.agent_index_list = [a.agent_id for a in self.AgentList]
        self.bid_count_list = [set() for _ in range(self.num_agents)]

    def solve(self, AgentList, TaskList, WorldInfoInput, max_depth, time_window_flag, graph=None):
        self.settings(AgentList, TaskList, WorldInfoInput, max_depth, time_window_flag)
        if graph is not None:
            self.graph = graph
        iter_idx = 1
        time_mat = [[0] * self.num_agents for _ in range(self.num_agents)]
        iter_prev = 0
        done_flag = False
        max_iter = 1000
        while not done_flag:
            if iter_idx > max_iter:
                break
            time_mat = self.communicate(time_mat, iter_idx)
            for idx_agent in range(self.num_agents):
                new_bid_flag = self.bundle(idx_agent)
                if new_bid_flag:
                    iter_prev = iter_idx
            if (iter_idx - iter_prev) > self.num_agents:
                done_flag = True
            elif (iter_idx - iter_prev) > (2 * self.num_agents):
                done_flag = True
            else:
                iter_idx += 1
        self.consensus_rounds = iter_idx

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
        self.path_list = [list(filter(lambda a: a != -1, self.path_list[i])) for i in range(len(self.path_list))]
        self.times_list = [list(filter(lambda a: a != -1, self.times_list[i])) for i in range(len(self.times_list))]
        return self.path_list, self.times_list

    def bundle(self, idx_agent):
        self.bundle_remove(idx_agent)
        new_bid_flag = self.bundle_add(idx_agent)
        return new_bid_flag

    def bundle_remove(self, idx_agent):
        out_bid_for_task = False
        for idx in range(self.max_depth):
            if self.bundle_list[idx_agent][idx] < 0:
                break
            else:
                if self.winners_list[idx_agent][self.bundle_list[idx_agent][idx]] != self.agent_index_list[idx_agent]:
                    out_bid_for_task = True
                if out_bid_for_task:
                    if self.winners_list[idx_agent][self.bundle_list[idx_agent][idx]] == self.agent_index_list[idx_agent]:
                        self.winners_list[idx_agent][self.bundle_list[idx_agent][idx]] = -1
                        self.winner_bid_list[idx_agent][self.bundle_list[idx_agent][idx]] = -1
                    path_current = copy.deepcopy(self.path_list[idx_agent])
                    if self.bundle_list[idx_agent][idx] in path_current:
                        idx_remove = path_current.index(self.bundle_list[idx_agent][idx])
                        del self.path_list[idx_agent][idx_remove]
                        self.path_list[idx_agent].append(-1)
                        del self.times_list[idx_agent][idx_remove]
                        self.times_list[idx_agent].append(-1)
                        del self.scores_list[idx_agent][idx_remove]
                        self.scores_list[idx_agent].append(-1)
                    self.bundle_list[idx_agent][idx] = -1

    def bundle_add(self, idx_agent):
        epsilon = 1e-5
        new_bid_flag = False
        index_array = np.where(np.array(self.bundle_list[idx_agent]) == -1)[0]
        bundle_full_flag = len(index_array) == 0
        feasibility = [[1] * (self.max_depth + 1) for _ in range(self.num_tasks)]
        alpha = 0.5
        max_tasks = self.max_depth if self.max_depth > 0 else 1
        current_load = len([t for t in self.path_list[idx_agent] if t != -1])
        load_penalty = 1 - alpha * (current_load / max_tasks)
        while not bundle_full_flag:
            best_indices, task_times, feasibility = self.compute_bid(idx_agent, feasibility)
            bid_list_balanced = [b * load_penalty if b > 0 else b for b in self.bid_list[idx_agent]]
            array_logical_1 = ((np.array(bid_list_balanced) - np.array(self.winner_bid_list[idx_agent])) > epsilon)
            array_logical_2 = (abs(np.array(bid_list_balanced) - np.array(self.winner_bid_list[idx_agent])) <= epsilon)
            array_logical_3 = (self.agent_index_list[idx_agent] < np.array(self.winners_list[idx_agent]))
            array_logical_result = np.logical_or(array_logical_1, np.logical_and(array_logical_2, array_logical_3))
            array_max = np.array(bid_list_balanced) * array_logical_result
            best_task = array_max.argmax()
            value_max = max(array_max)
            if value_max > 0:
                new_bid_flag = True
                all_values = np.where(array_max == value_max)[0]
                if len(all_values) == 1:
                    best_task = all_values[0]
                else:
                    earliest = float('inf')
                    for i in range(len(all_values)):
                        if self.TaskList[all_values[i]].start_time < earliest:
                            earliest = self.TaskList[all_values[i]].start_time
                            best_task = all_values[i]
                self.winners_list[idx_agent][best_task] = self.AgentList[idx_agent].agent_id
                self.winner_bid_list[idx_agent][best_task] = self.bid_list[idx_agent][best_task]
                self.path_list[idx_agent].insert(best_indices[best_task], best_task)
                del self.path_list[idx_agent][-1]
                self.times_list[idx_agent].insert(best_indices[best_task], task_times[best_task])
                del self.times_list[idx_agent][-1]
                self.scores_list[idx_agent].insert(best_indices[best_task], bid_list_balanced[best_task])
                del self.scores_list[idx_agent][-1]
                length = len(np.where(np.array(self.bundle_list[idx_agent]) > -1)[0])
                self.bundle_list[idx_agent][length] = best_task
                for i in range(self.num_tasks):
                    feasibility[i].insert(best_indices[best_task], feasibility[i][best_indices[best_task]])
                    del feasibility[i][-1]
                self.bid_count_list[idx_agent].add(best_task)
            else:
                break
            index_array = np.where(np.array(self.bundle_list[idx_agent]) == -1)[0]
            bundle_full_flag = len(index_array) == 0
        return new_bid_flag

    def communicate(self, time_mat, iter_idx):
        time_mat_new = copy.deepcopy(time_mat)
        old_z = copy.deepcopy(self.winners_list)
        old_y = copy.deepcopy(self.winner_bid_list)
        z = copy.deepcopy(old_z)
        y = copy.deepcopy(old_y)
        epsilon = 10e-6
        comm_this_round = 0
        for k in range(self.num_agents):
            for i in range(self.num_agents):
                if self.graph[k][i] == 1:
                    comm_this_round += 1
                    for j in range(self.num_tasks):
                        if old_z[k][j] == k:
                            if z[i][j] == i:
                                if (old_y[k][j] - y[i][j]) > epsilon:
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                elif abs(old_y[k][j] - y[i][j]) <= epsilon:
                                    if z[i][j] > old_z[k][j]:
                                        z[i][j] = old_z[k][j]
                                        y[i][j] = old_y[k][j]
                            elif z[i][j] == k:
                                z[i][j] = old_z[k][j]
                                y[i][j] = old_y[k][j]
                            elif z[i][j] > -1 and z[i][j] < self.num_agents:
                                if time_mat[k][z[i][j]] > time_mat_new[i][z[i][j]]:
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                elif (old_y[k][j] - y[i][j]) > epsilon:
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                elif abs(old_y[k][j] - y[i][j]) <= epsilon:
                                    if z[i][j] > old_z[k][j]:
                                        z[i][j] = old_z[k][j]
                                        y[i][j] = old_y[k][j]
                            elif z[i][j] == -1:
                                z[i][j] = old_z[k][j]
                                y[i][j] = old_y[k][j]
                        elif old_z[k][j] == i:
                            if z[i][j] == i:
                                pass
                            elif z[i][j] == k:
                                z[i][j] = -1
                                y[i][j] = -1
                            elif z[i][j] > -1 and z[i][j] < self.num_agents:
                                if time_mat[k][z[i][j]] > time_mat_new[i][z[i][j]]:
                                    z[i][j] = -1
                                    y[i][j] = -1
                            elif z[i][j] == -1:
                                pass
                        elif old_z[k][j] > -1 and old_z[k][j] < self.num_agents:
                            if z[i][j] == i:
                                if time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]:
                                    if (old_y[k][j] - y[i][j]) > epsilon:
                                        z[i][j] = old_z[k][j]
                                        y[i][j] = old_y[k][j]
                                    elif abs(old_y[k][j] - y[i][j]) <= epsilon:
                                        if z[i][j] > old_z[k][j]:
                                            z[i][j] = old_z[k][j]
                                            y[i][j] = old_y[k][j]
                            elif z[i][j] == k:
                                if time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]:
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                else:
                                    z[i][j] = -1
                                    y[i][j] = -1
                            elif z[i][j] == old_z[k][j]:
                                if time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]:
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                            elif z[i][j] > -1 and z[i][j] < self.num_agents:
                                if time_mat[k][z[i][j]] > time_mat_new[i][z[i][j]]:
                                    if time_mat[k][old_z[k][j]] >= time_mat_new[i][old_z[k][j]]:
                                        z[i][j] = old_z[k][j]
                                        y[i][j] = old_y[k][j]
                                    elif time_mat[k][old_z[k][j]] < time_mat_new[i][old_z[k][j]]:
                                        z[i][j] = -1
                                        y[i][j] = -1
                                else:
                                    if time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]:
                                        if (old_y[k][j] - y[i][j]) > epsilon:
                                            z[i][j] = old_z[k][j]
                                            y[i][j] = old_y[k][j]
                                        elif abs(old_y[k][j] - y[i][j]) <= epsilon:
                                            if z[i][j] > old_z[k][j]:
                                                z[i][j] = old_z[k][j]
                                                y[i][j] = old_y[k][j]
                            elif z[i][j] == -1:
                                if time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]:
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                        elif old_z[k][j] == -1:
                            if z[i][j] == i:
                                pass
                            elif z[i][j] == k:
                                z[i][j] = old_z[k][j]
                                y[i][j] = old_y[k][j]
                            elif z[i][j] > -1 and z[i][j] < self.num_agents:
                                if time_mat[k][z[i][j]] > time_mat_new[i][z[i][j]]:
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                            elif z[i][j] == -1:
                                pass
                    for n in range(self.num_agents):
                        if (n != i) and (time_mat_new[i][n] < time_mat[k][n]):
                            time_mat_new[i][n] = time_mat[k][n]
                    time_mat_new[i][k] = iter_idx
        self.comm_count += comm_this_round
        self.winners_list = copy.deepcopy(z)
        self.winner_bid_list = copy.deepcopy(y)
        return time_mat_new

    def compute_bid(self, idx_agent, feasibility):
        empty_task_index_list = np.where(np.array(self.path_list[idx_agent]) == -1)[0]
        if len(empty_task_index_list) == 0:
            return [], [], feasibility
        self.bid_list[idx_agent] = [-1] * self.num_tasks
        best_indices = [-1] * self.num_tasks
        task_times = [-2] * self.num_tasks
        alpha = 1.2
        max_tasks = self.max_depth if self.max_depth > 0 else 1
        current_load = len([t for t in self.path_list[idx_agent] if t != -1])
        max_per_agent = int(np.ceil(self.num_tasks / self.num_agents))
        for idx_task in range(self.num_tasks):
            if self.compatibility_mat[self.AgentList[idx_agent].agent_type][self.TaskList[idx_task].task_type] > 0.5:
                index_array = np.where(np.array(self.path_list[idx_agent][0:empty_task_index_list[0]]) == idx_task)[0]
                if len(index_array) < 0.5:
                    best_bid = 0
                    best_index = -1
                    best_time = -2
                    for j in range(empty_task_index_list[0] + 1):
                        if feasibility[idx_task][j] == 1:
                            skip_flag = False
                            if j == 0:
                                task_prev = []
                                time_prev = []
                            else:
                                Task_temp = self.TaskList[self.path_list[idx_agent][j - 1]]
                                task_prev = Task(**Task_temp.__dict__)
                                time_prev = self.times_list[idx_agent][j - 1]
                            if j == (empty_task_index_list[0]):
                                task_next = []
                                time_next = []
                            else:
                                Task_temp = self.TaskList[self.path_list[idx_agent][j]]
                                task_next = Task(**Task_temp.__dict__)
                                time_next = self.times_list[idx_agent][j]
                            Task_temp = self.TaskList[idx_task]
                            [score, min_start, max_start] = self.scoring_compute_score(
                                idx_agent, Task(**Task_temp.__dict__), task_prev, time_prev, task_next, time_next
                            )
                            if current_load >= max_per_agent:
                                load_penalty = 0.0
                            else:
                                load_penalty = np.exp(-alpha * current_load)
                            score = score * load_penalty
                            window_start = Task_temp.start_time if hasattr(Task_temp, 'window_start') else 0
                            window_end = Task_temp.end_time if hasattr(Task_temp, 'window_end') else 1000
                            if min_start < window_start or (min_start + Task_temp.duration) > window_end:
                                skip_flag = True
                                feasibility[idx_task][j] = 0
                            if not skip_flag:
                                if score > best_bid:
                                    best_bid = score
                                    best_index = j
                                    best_time = min_start
                    if best_bid > 0:
                        self.bid_list[idx_agent][idx_task] = best_bid
                        best_indices[idx_task] = best_index
                        task_times[idx_task] = best_time
        return best_indices, task_times, feasibility

    def scoring_compute_score(self, idx_agent, task_current, task_prev, time_prev, task_next, time_next):
        if self.AgentList[idx_agent].agent_type in [0, 1, 2]:
            if not task_prev:
                dt = math.sqrt((self.AgentList[idx_agent].x - task_current.x)**2 +
                               (self.AgentList[idx_agent].y - task_current.y)**2 +
                               (self.AgentList[idx_agent].z - task_current.z)**2) / \
                     self.AgentList[idx_agent].nom_velocity
                min_start = max(task_current.start_time, self.AgentList[idx_agent].availability + dt)
            else:
                dt = math.sqrt((task_prev.x - task_current.x)**2 +
                               (task_prev.y - task_current.y)**2 +
                               (task_prev.z - task_current.z)**2) / \
                     self.AgentList[idx_agent].nom_velocity
                min_start = max(task_current.start_time, time_prev + task_prev.duration + dt)
            if not task_next:
                dt = 0.0
                max_start = task_current.end_time
            else:
                dt = math.sqrt((task_next.x - task_current.x)**2 +
                               (task_next.y - task_current.y)**2 +
                               (task_next.z - task_current.z)**2) / \
                     self.AgentList[idx_agent].nom_velocity
                max_start = min(task_current.end_time, time_next - task_current.duration - dt)
            if self.time_window_flag:
                reward = task_current.task_value * math.exp(
                    (-task_current.discount) * (min_start - task_current.start_time)
                )
            else:
                dt_current = math.sqrt((self.AgentList[idx_agent].x - task_current.x)**2 +
                                       (self.AgentList[idx_agent].y - task_current.y)**2 +
                                       (self.AgentList[idx_agent].z - task_current.z)**2) / \
                             self.AgentList[idx_agent].nom_velocity
                reward = task_current.task_value * math.exp(
                    (-task_current.discount) * dt_current
                )
            score = reward
        else:
            raise Exception("Неизвестный тип агента!")
        return score, min_start, max_start

    def get_bid_count_list(self):
        return [len(s) for s in self.bid_count_list]


def dicts_to_agents(uavs, agent_types):
    type_map = {name: idx for idx, name in enumerate(agent_types)}
    agents = []
    for u in uavs:
        agents.append(Agent(
            agent_id=u['id'],
            agent_type=type_map.get(u['type'], 0),
            x=u['pos'][0],
            y=u['pos'][1],
            z=0,
            nom_velocity=10,
            availability=0,
            fuel=u.get('fuel', 0)
        ))
    return agents


def dicts_to_tasks(tasks, task_types):
    type_map = {name: idx for idx, name in enumerate(task_types)}
    task_objs = []
    for t in tasks:
        task_objs.append(Task(
            task_id=t['id'],
            task_type=type_map.get(t['type'], 0),
            x=t['location'][0],
            y=t['location'][1],
            z=0,
            start_time=t.get('start_time', 0),
            end_time=t.get('end_time', 1000),
            duration=t.get('duration', 1),
            task_value=t.get('task_value', 10),
            discount=t.get('discount', 0.01)
        ))
    return task_objs
