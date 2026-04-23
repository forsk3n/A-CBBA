"""
A-CBBA: Adaptive Consensus Stopping + Weighted Relay Selection
Модификация CBBA для работы в условиях неидеальных каналов связи.

задача
  Базовый CBBA использует фиксированное число раундов консенсуса, определяемое
  диаметром полностью связного графа. В каналах с потерями это приводит к
  преждевременной остановке (сбой сходимости), а в идеальных каналах —
  к избыточным раундам (потеря пропускной способности).
решение
  1. Адаптивный останов: каждый агент отслеживает стабильность локальной таблицы
     ставок на протяжении τ последовательных раундов; глобальная остановка,
     когда все достижимые агенты стабильны.
  2. Взвешенный выбор ретранслятора: оценка агента-ретранслятора по формуле
       score(k,d) = α·R_k + (1-α)·Q_{k,d}
     где R_k = надёжность (заряд батареи), Q_{k,d} = качество канала.
"""

import numpy as np
import math
from Depencies.cbba_core import CBBA

# ─────────────────────────────────────────────────────────────────
#  Вспомогательная функция: применение бернуллиевских потерь пакетов
# ─────────────────────────────────────────────────────────────────

def apply_packet_loss(adj_matrix, p_loss, rng=None, symmetric=True):
    """
    Возвращает зашумлённую копию матрицы смежности: каждое ребро удаляется
    с вероятностью p_loss, при несимметричных потерях - Возвращает матрицу смежности с потерями, потери независимы для (i,j) и (j,i).
    Параметры:
        adj_matrix : исходная матрица смежности (list of lists или np.array)
        p_loss     : вероятность потери пакета на ребре (0..1)
        rng        : генератор случайных чисел (для воспроизводимости)
    Возвращает:
        матрицу смежности с потерянными рёбрами (симметричная)
    """
    if rng is None:
        rng = np.random.default_rng()
    adj = np.array(adj_matrix, dtype=int).copy()
    n = adj.shape[0]
    np.fill_diagonal(adj, 0)                    # обнуляем диагональ
    # Генерируем маску для каждого направления
    mask = rng.random((n, n)) > p_loss
    if symmetric:
        mask = np.triu(mask, 1)  # работаем только с верхним треугольником
        mask = mask | mask.T        # делаем симметричной
    else:
        # Оставляем как есть
        np.fill_diagonal(mask, 0)
    adj = (adj & mask).astype(int)  # применяем маску
    return adj.tolist()

# ─────────────────────────────────────────────────────────────────
#  A-CBBA: подкласс CBBA с адаптивной остановкой
# ─────────────────────────────────────────────────────────────────
class ACBBA(CBBA):
    """
    Расширяет CBBA следующими возможностями:
      - симуляция потерь пакетов на каждом раунде консенсуса
      - адаптивная остановка на основе стабильности таблицы ставок (окно τ)
      - взвешенный выбор агента-ретранслятора (надёжность × качество канала)
    """

    def __init__(self, agent_types, task_types, compatibility_mat,
                 p_loss=0.0, tau=3, alpha_relay=0.5, max_extra_rounds=30, arena=1000.0):
        """
        Конструктор A-CBBA.
        Параметры:
            agent_types      : список типов агентов
            task_types       : список типов задач
            compatibility_mat: матрица совместимости агент-задача
            p_loss           : вероятность потери пакета на ребре (по умолчанию 0)
            tau              : размер окна стабильности (в раундах)
            alpha_relay      : вес надёжности при выборе ретранслятора (0..1)
            max_extra_rounds : максимальное число дополнительных раундов (защита)
        """
        super().__init__(agent_types, task_types, compatibility_mat)
        self.p_loss        = p_loss          # вероятность потери пакета
        self.tau           = tau             # окно стабильности (раунды)
        self.alpha_relay   = alpha_relay     # вес надёжности vs качества канала
        self.max_extra_rounds = max_extra_rounds
        self.arena = arena
        self._rng = np.random.default_rng(seed=42)   # воспроизводимость результата

    # ── коэффициент надёжности (прокси: нормализованный уровень топлива) ────
    def _reliability(self, agent_idx):
        """
        Вычисляет динамический коэффициент надёжности агента.
        Используется: нормализованный запас топлива.
        Параметры:
            agent_idx: индекс агента в списке AgentList
        Возвращает:
            значение надёжности в диапазоне [0, 1] (1 — максимальная надёжность)
        """
        fuel = self.AgentList[agent_idx].fuel
        max_fuel = max((a.fuel for a in self.AgentList), default=1) or 1
        return float(fuel) / max_fuel

    # ── качество канала между двумя агентами (на основе расстояния) ───────
    def _channel_quality(self, src_idx, dst_idx):
        """
        Оценивает качество прямого канала связи между двумя агентами.
        Моделируется как функция от расстояния: чем ближе, тем выше качество.
        Параметры:
            src_idx : индекс источника
            dst_idx : индекс получателя
        Возвращает:
            значение качества в диапазоне [0, 1]
        """
        a = self.AgentList[src_idx]
        b = self.AgentList[dst_idx]
        # Евклидово расстояние между агентами
        dist = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        max_dist = self.arena * math.sqrt(2)  # диагональ квадратной арены
        return max(0.0, 1.0 - dist / max_dist)

    # ── оценка ретранслятора ───────────────────────────────────────────────
    def _relay_score(self, src_idx, relay_idx, dst_idx):
        """
        Вычисляет комбинированную оценку агента-ретранслятора для передачи
        сообщения от src_idx к dst_idx через relay_idx.
        Формула: score = α * R + (1-α) * Q, где
        R = надёжность ретранслятора,
        Q = min( Q(src, relay) * Q(relay, dst) )
        Параметры:
            relay_idx : индекс потенциального ретранслятора
            dst_idx   : индекс конечного получателя
        Возвращает:
            числовую оценку (чем выше, тем лучше ретранслятор)
        """
        R = self._reliability(relay_idx)                    #Качество канала от источника к ретранслятору
        Q_ik = self._channel_quality(src_idx, relay_idx)    #Качество канала от ретранслятора к получателю
        Q_kj = self._channel_quality(relay_idx, dst_idx)     #Комбинированное качество
        Q = min(Q_ik, Q_kj)
        return self.alpha_relay * R + (1.0 - self.alpha_relay) * Q

    # ── построение эффективного графа на раунд с потерями и ретрансляцией ──
    def _effective_graph(self, base_graph):
        n = self.num_agents
        lossy = apply_packet_loss(base_graph, self.p_loss, self._rng, symmetric=False) #делаем асимметричные потери
        lossy_np = np.array(lossy, dtype=int)

        effective = lossy_np.copy()
        # Перебираем только пары i < j
        for i in range(n):
            for j in range(i + 1, n):
                if effective[i][j] == 1:
                    continue  # уже есть прямая связь
                best_score = -1.0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if lossy_np[i][k] == 1 and lossy_np[k][j] == 1:
                        s = self._relay_score(i, k, j)
                        if s > best_score:
                            best_score = s
                if best_score > 0.3:
                    effective[i][j] = 1
                    effective[j][i] = 1  # симметрично
        np.fill_diagonal(effective, 0)
        return effective.tolist()

    # ── адаптивная проверка сходимости ─────────────────────────────────
    def _check_stability(self, history):
        """
        Проверяет, стабильны ли таблицы winner_bid_list для всех агентов
        на протяжении последних τ раундов.
        Параметры:
            history : список снимков winner_bid_list (новейший в конце)
        Возвращает:
            True, если все таблицы не менялись последние τ раундов, иначе False
        """
        if len(history) < self.tau:
            return False
        recent = history[-self.tau:]   # берём последние τ снимков
        for agent in range(self.num_agents):
            for r in range(1, self.tau):
                # сравниваем текущий снимок с предыдущим для этого агента
                if recent[r][agent] != recent[r-1][agent]:
                    return False
        return True

    # ── переопределение метода solve() ──────────────────────────────────────────
    def solve(self, AgentList, TaskList, WorldInfoInput, max_depth,
              time_window_flag, graph=None):
        """
        Основной метод решения с адаптивной остановкой.
        Выполняет итеративные раунды: построение эффективного графа, коммуникация,
        обновление bundles, проверка стабильности.
        Возвращает:
            path_list  : список маршрутов для каждого агента (в ID задач)
            times_list : список времен начала задач для каждого агента
        """
        # Инициализация структур данных (унаследовано от CBBA)
        self.settings(AgentList, TaskList, WorldInfoInput,
                      max_depth, time_window_flag)
        base_graph = graph if graph is not None else self.graph
        base_graph_np = np.array(base_graph)

        iter_idx = 1
        time_mat = [[0]*self.num_agents for _ in range(self.num_agents)]
        bid_history = []        # история снимков winner_bid_list для адаптивной остановки
        done_flag  = False
        max_iter   = 500

        while not done_flag:
            if iter_idx > max_iter:
                break

            # 1. Строим эффективный граф для текущего раунда (с потерями и ретрансляцией)
            eff_graph = self._effective_graph(base_graph_np.tolist())
            self.graph = eff_graph

            # 2. Выполняем один раунд коммуникации (обмен победителями и ставками)
            time_mat = self.communicate(time_mat, iter_idx)

            # 3. Каждый агент обновляет свой bundle (жадное добавление/удаление задач)
            for idx_agent in range(self.num_agents):
                self.bundle(idx_agent)

            # 4. Сохраняем снимок текущих таблиц winner_bid_list для всех агентов
            snapshot = [list(self.winner_bid_list[a]) for a in range(self.num_agents)]
            bid_history.append(snapshot)

            # 5. Проверяем стабильность на протяжении τ раундов
            if self._check_stability(bid_history) and iter_idx > 1:
                done_flag = True
            else:
                iter_idx += 1

        # Сохраняем число раундов консенсуса для метрик (полезно для экспериментов)
        self.consensus_rounds = iter_idx

        # Постобработка: преобразование индексов задач в их ID (аналогично базовому CBBA)
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

        # Удаляем все -1 (пустые слоты) из маршрутов и времен
        self.path_list  = [list(filter(lambda a: a != -1, self.path_list[i]))  for i in range(len(self.path_list))]
        self.times_list = [list(filter(lambda a: a != -1, self.times_list[i])) for i in range(len(self.times_list))]
        return self.path_list, self.times_list
