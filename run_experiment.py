"""
Эксперимент для сравнения: CBBA против A-CBBA при различных потерях пакетов.
Результаты:
  - results/tcr_vs_ploss.png        (доля выполненных задач)
  - results/rounds_vs_ploss.png     (раунды консенсуса)
  - results/conflict_vs_ploss.png   (доля бесконфликтных назначений)
  - results/summary_table.csv       (все метрики)
"""
import copy
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from Depencies.cbba_core  import CBBA, WorldInfo, Agent, Task
from acbba_core import ACBBA, apply_packet_loss

os.makedirs('results', exist_ok=True)

# ── Параметры эксперимента ─────────────────────────────────────────
N_RUNS   = 50            # число прогонов Монте-Карло на одно условие
N_UAV    = 10            # размер роя для экспериментов
N_TASKS  = 20
P_LOSS_LEVELS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
MAX_DEPTH     = 5        # максимальное число задач на одного агента
SEED          = 0

# ── Генерация воспроизводимого сценария ──────────────────────────
rng_global = np.random.default_rng(SEED)

def build_scenario(rng, n_uav=N_UAV, n_tasks=N_TASKS, arena=1000.0):
    """
    Создаёт случайный сценарий: агенты и задачи с координатами,
    временными окнами и топливом.
    """
    agents = []
    for i in range(n_uav):
        x = float(rng.integers(0, int(arena)))
        y = float(rng.integers(0, int(arena)))
        fuel = float(rng.integers(500, 1001))
        agents.append(Agent(
            agent_id=i, agent_type=0,
            x=x, y=y, z=0.0,
            nom_velocity=10.0,
            availability=0.0,
            fuel=fuel
        ))
    tasks = []
    for j in range(n_tasks):
        x = float(rng.integers(0, int(arena)))
        y = float(rng.integers(0, int(arena)))
        start = float(rng.integers(0, 800))
        end   = start + float(rng.integers(100, 300))
        tasks.append(Task(
            task_id=j, task_type=0,
            x=x, y=y, z=0.0,
            start_time=start, end_time=end,
            duration=10.0, task_value=10.0, discount=0.005
        ))
    world = WorldInfo(arena, arena, 100.0)
    compat = [[1]]   # агент типа 0 может выполнять задачу типа 0
    return agents, tasks, world, compat

def random_geometric_graph(n, radius=0.5, arena=1000.0, rng=None):
    """
    Случайный геометрический граф: ребро существует, если расстояние
    между агентами < radius * arena.
    """
    if rng is None:
        rng = np.random.default_rng()
    pos = rng.random((n, 2)) * arena
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < radius * arena:
                adj[i,j] = adj[j,i] = 1
    np.fill_diagonal(adj, 0)
    return adj

# ── Вспомогательные функции для метрик ───────────────────────────────
def task_completion_rate(path_list, n_tasks):
    """Доля задач, назначенных хотя бы одному агенту."""
    assigned = set()
    for path in path_list:
        assigned.update(path)
    return len(assigned) / n_tasks

def count_conflicts(path_list):
    """Количество конфликтов (задача назначена нескольким агентам)."""
    seen, conflicts = set(), 0
    for path in path_list:
        for t in path:
            if t in seen:
                conflicts += 1
            seen.add(t)
    return conflicts

def conflict_free_rate(path_list):
    """1.0, если ни одна задача не назначена более чем одному агенту, иначе 0.0."""
    seen, total = set(), 0
    conflict = False
    for path in path_list:
        for t in path:
            total += 1
            if t in seen:
                conflict = True
            seen.add(t)
    return 0.0 if conflict else 1.0

# ── Запуск одного эксперимента для CBBA с заданными потерями ──────────
def run_cbba(agents, tasks, world, compat, base_adj, p_loss, rng):
    """
    Запускает базовый CBBA на графе с потерями пакетов.
    Возвращает (path_list, число раундов консенсуса).
    """
    cbba = CBBA(['type0'], ['type0'], compat)
    # Используем переданный rng для потерь пакетов
    lossy_adj = apply_packet_loss(base_adj, p_loss, rng)
    path, times = cbba.solve(
        copy.deepcopy(agents), copy.deepcopy(tasks), world,
        MAX_DEPTH, False, graph=lossy_adj
    )
    rounds = cbba.consensus_rounds
    return path, rounds

# ── Запуск одного эксперимента для A-CBBA ────────────────────────────
def run_acbba(agents, tasks, world, compat, base_adj, p_loss, rng):
    """
    Запускает A-CBBA (адаптивный останов + ретрансляция) на исходном графе.
    Потери моделируются внутри A-CBBA на каждом раунде.
    Возвращает (path_list, число раундов консенсуса).
    """
    acbba = ACBBA(['type0'], ['type0'], compat, p_loss=p_loss, tau=3, arena=world.limit_x)
    acbba._rng = rng
    path, times = acbba.solve(
        copy.deepcopy(agents), copy.deepcopy(tasks), world,
        MAX_DEPTH, False, graph=base_adj
    )
    return path, acbba.consensus_rounds

# ── Основной цикл эксперимента ──────────────────────────────────────────
print("Запуск экспериментов...")
results = {
    'cbba':  {'tcr': [], 'rounds': [], 'cfr': []},
    'acbba': {'tcr': [], 'rounds': [], 'cfr': []},
}

for p_loss in P_LOSS_LEVELS:
    cbba_tcr, cbba_rnd, cbba_cfr   = [], [], []
    acbba_tcr, acbba_rnd, acbba_cfr = [], [], []

    for run in range(N_RUNS):
        # Генератор с уникальным seed для воспроизводимости
        rng = np.random.default_rng(SEED * 1000 + int(p_loss*100) * 100 + run)
        agents, tasks, world, compat = build_scenario(rng)
        base_adj = random_geometric_graph(N_UAV, radius=0.45, rng=rng)

        # Гарантируем минимальную связность (если граф пуст — делаем полный)
        if np.sum(base_adj) == 0:
            base_adj = (np.ones((N_UAV,N_UAV)) - np.eye(N_UAV)).astype(int)

        # --- CBBA ---
        path_c, rnd_c = run_cbba(agents, tasks, world, compat, base_adj, p_loss, np.random.default_rng(run))
        cbba_tcr.append(task_completion_rate(path_c, N_TASKS))
        cbba_rnd.append(rnd_c)
        cbba_cfr.append(conflict_free_rate(path_c))

        # --- A-CBBA ---
        path_a, rnd_a = run_acbba(agents, tasks, world, compat, base_adj, p_loss, np.random.default_rng(run+500))
        acbba_tcr.append(task_completion_rate(path_a, N_TASKS))
        acbba_rnd.append(rnd_a)
        acbba_cfr.append(conflict_free_rate(path_a))

    # Усреднение по прогонам
    results['cbba']['tcr'].append(np.mean(cbba_tcr))
    results['cbba']['rounds'].append(np.mean(cbba_rnd))
    results['cbba']['cfr'].append(np.mean(cbba_cfr))
    results['acbba']['tcr'].append(np.mean(acbba_tcr))
    results['acbba']['rounds'].append(np.mean(acbba_rnd))
    results['acbba']['cfr'].append(np.mean(acbba_cfr))

    print(f"  p_loss={p_loss:.0%}  CBBA: доля задач={np.mean(cbba_tcr):.2%} раундов={np.mean(cbba_rnd):.1f}  |  "
          f"A-CBBA: доля задач={np.mean(acbba_tcr):.2%} раундов={np.mean(acbba_rnd):.1f}")

# ── Настройка стиля графиков ─────────────────────────────────────────
STYLE = {
    'cbba':  dict(color='#2166AC', marker='s', ls='-',  lw=1.8, label='Базовый CBBA'),
    'acbba': dict(color='#D73027', marker='^', ls='--', lw=1.8, label='A-CBBA'),
}
xs = [int(p*100) for p in P_LOSS_LEVELS]

def make_fig():
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.tick_params(labelsize=9)
    ax.set_xlabel('Вероятность потери пакетов (%)', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    return fig, ax

# График 1: Доля выполненных задач
fig, ax = make_fig()
for alg in ('cbba', 'acbba'):
    ax.plot(xs, [v*100 for v in results[alg]['tcr']], **STYLE[alg])
ax.set_ylabel('Доля выполненных задач (%)', fontsize=10)
ax.set_title('Доля выполненных задач в зависимости от потерь пакетов', fontsize=10)
ax.set_ylim(0, 105)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig('results/tcr_vs_ploss.png', dpi=150)
plt.close(fig)

# График 2: Раунды консенсуса
fig, ax = make_fig()
for alg in ('cbba', 'acbba'):
    ax.plot(xs, results[alg]['rounds'], **STYLE[alg])
ax.set_ylabel('Раунды консенсуса (средн.)', fontsize=10)
ax.set_title('Раунды консенсуса в зависимости от потерь пакетов', fontsize=10)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig('results/rounds_vs_ploss.png', dpi=150)
plt.close(fig)

# График 3: Комбинированный (две панели)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, metric, ylabel, title in zip(
    axes,
    [('tcr', 100, '%'), ('rounds', 1, '')],
    ['Доля выполненных задач (%)', 'Раунды консенсуса (средн.)'],
    ['(a) Доля выполненных задач', '(b) Раунды консенсуса'],
):
    key, scale, unit = metric
    for alg in ('cbba', 'acbba'):
        ax.plot(xs, [v*scale for v in results[alg][key]], **STYLE[alg])
    ax.set_xlabel('Потери пакетов (%)', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
fig.tight_layout()
fig.savefig('results/combined_figure.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Сохранение CSV-таблицы ─────────────────────────────────────────────
import csv
with open('results/summary_table.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['p_loss_%', 'cbba_tcr_%', 'cbba_rounds', 'cbba_cfr_%',
                'acbba_tcr_%', 'acbba_rounds', 'acbba_cfr_%'])
    for i, p in enumerate(P_LOSS_LEVELS):
        w.writerow([
            int(p*100),
            f"{results['cbba']['tcr'][i]*100:.1f}",
            f"{results['cbba']['rounds'][i]:.1f}",
            f"{results['cbba']['cfr'][i]*100:.1f}",
            f"{results['acbba']['tcr'][i]*100:.1f}",
            f"{results['acbba']['rounds'][i]:.1f}",
            f"{results['acbba']['cfr'][i]*100:.1f}",
        ])

print("\nРезультаты сохранены в папку results/")
print("Содержимое CSV:")
with open('results/summary_table.csv') as f:
    print(f.read())
