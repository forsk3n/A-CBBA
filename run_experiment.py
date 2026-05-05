"""
Эксперимент для сравнения алгоритмов: CBBA и A-CBBA при различных потерях пакетов.

Статистическая обработка результатов:
  - mean, медиана, std, IQR, 95% доверительные интервалы по N_RUNS прогонам
  - парный t-критерий Стьюдента на каждом уровне p_loss
  - парный критерий знаковых рангов Уилкоксона
  - коэффициент размера эффекта Cohen's d (практическая значимость)
  - коррекция Бонферрони на множественные сравнения (6 уровней p_loss)
  - двухфакторный ANOVA с проверкой эффекта взаимодействия algorithm × p_loss
  - коэффициент Спирмена ρ(TCR, p_loss) — мера деградации каждого алгоритма
  - скорость деградации TCR: абсолютная (п.п.) и относительная (%) от базового уровня

Результаты:
  - results/tcr_vs_ploss.png        (TCR + 95% CI)
  - results/rounds_vs_ploss.png     (раунды + 95% CI)
  - results/cfr_vs_ploss.png        (CFR + 95% CI)
  - results/combined_figure.png     (две панели)
  - results/summary_table.csv       (mean, median, std, IQR, 95% CI)
  - results/statistical_tests.csv   (t-тест, Уилкоксон, Cohen's d, Бонферрони)
  - results/anova_results.txt       (двухфакторный ANOVA)
  - results/degradation.csv         (Spearman ρ, абс. и отн. деградация TCR)
  - results/raw_runs.csv            (все прогоны для воспроизводимости)
  - results/performance.txt         (замеры времени и ресурсов)
"""
import copy
import os
import sys
import csv
import time
import platform
import tracemalloc

import numpy as np
import matplotlib.pyplot as plt
import psutil
from scipy import stats

# Для двухфакторного ANOVA
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

sys.path.insert(0, os.path.dirname(__file__))
from Depencies.cbba_core import CBBA, WorldInfo, Agent, Task
from acbba_core import ACBBA, apply_packet_loss

os.makedirs('results', exist_ok=True)

# Параметры эксперимента
N_RUNS   = 50
N_UAV    = 10
N_TASKS  = 20
P_LOSS_LEVELS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
MAX_DEPTH     = 5
SEED          = 0
ALPHA         = 0.05
N_COMPARISONS = len(P_LOSS_LEVELS)
ALPHA_BONF    = ALPHA / N_COMPARISONS         # ≈ 0.0083

rng_global = np.random.default_rng(SEED)

# Генерация сценария
def build_scenario(rng, n_uav=N_UAV, n_tasks=N_TASKS, arena=1000.0):
    agents = []
    for i in range(n_uav):
        x = float(rng.integers(0, int(arena)))
        y = float(rng.integers(0, int(arena)))
        fuel = float(rng.integers(500, 1001))
        agents.append(Agent(
            agent_id=i, agent_type=0,
            x=x, y=y, z=0.0,
            nom_velocity=10.0, availability=0.0, fuel=fuel
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
    compat = [[1]]
    return agents, tasks, world, compat

def random_geometric_graph(n, radius=0.5, arena=1000.0, rng=None):
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

# Метрики
def task_completion_rate(path_list, n_tasks):
    assigned = set()
    for path in path_list:
        assigned.update(path)
    return len(assigned) / n_tasks

def conflict_free_rate(path_list):
    seen = set()
    conflict = False
    for path in path_list:
        for t in path:
            if t in seen:
                conflict = True
            seen.add(t)
    return 0.0 if conflict else 1.0

# Функции двух алгоритмов
def run_cbba(agents, tasks, world, compat, base_adj, p_loss, rng):
    cbba = CBBA(['type0'], ['type0'], compat)
    lossy_adj = apply_packet_loss(base_adj, p_loss, rng)
    path, times = cbba.solve(
        copy.deepcopy(agents), copy.deepcopy(tasks), world,
        MAX_DEPTH, False, graph=lossy_adj
    )
    return path, cbba.consensus_rounds

def run_acbba(agents, tasks, world, compat, base_adj, p_loss, rng):
    acbba = ACBBA(['type0'], ['type0'], compat, p_loss=p_loss, tau=3, arena=world.limit_x)
    acbba._rng = rng
    path, times = acbba.solve(
        copy.deepcopy(agents), copy.deepcopy(tasks), world,
        MAX_DEPTH, False, graph=base_adj
    )
    return path, acbba.consensus_rounds

# Информация о системе и инициализация замеров
print("="*78)
print("Информация о системе и окружении")
print("="*78)
print(f"  ОС:               {platform.system()} {platform.release()}")
print(f"  Архитектура:      {platform.machine()}")
print(f"  Процессор:        {platform.processor() or 'не определён'}")
print(f"  Логических ядер:  {os.cpu_count()}")
print(f"  Физических ядер:  {psutil.cpu_count(logical=False)}")
print(f"  ОЗУ всего:        {psutil.virtual_memory().total/1024**3:.1f} ГБ")
print(f"  Python:           {platform.python_version()}")
print(f"  NumPy:            {np.__version__}")
print(f"  SciPy:            {stats.__name__.split('.')[0]} {__import__('scipy').__version__}")
import statsmodels
print(f"  statsmodels:      {statsmodels.__version__}")
print()

# Старт замера ресурсов
tracemalloc.start()
T_total_start = time.perf_counter()
T_cpu_start   = time.process_time()
proc = psutil.Process()
cpu_t0 = proc.cpu_times()

# Основной цикл
print("Запуск экспериментов.")
print(f"Параметры: N_UAV={N_UAV}, N_TASKS={N_TASKS}, N_RUNS={N_RUNS}, "
      f"уровней p_loss={len(P_LOSS_LEVELS)}, всего запусков={2*len(P_LOSS_LEVELS)*N_RUNS}\n")

raw = {
    'cbba':  {'tcr': [], 'rounds': [], 'cfr': [], 'time_ms': []},
    'acbba': {'tcr': [], 'rounds': [], 'cfr': [], 'time_ms': []},
}

for p_loss in P_LOSS_LEVELS:
    cbba_tcr, cbba_rnd, cbba_cfr, cbba_t   = [], [], [], []
    acbba_tcr, acbba_rnd, acbba_cfr, acbba_t = [], [], [], []

    for run in range(N_RUNS):
        rng = np.random.default_rng(SEED * 1000 + int(p_loss*100) * 100 + run)
        agents, tasks, world, compat = build_scenario(rng)
        base_adj = random_geometric_graph(N_UAV, radius=0.45, rng=rng)
        if np.sum(base_adj) == 0:
            base_adj = (np.ones((N_UAV,N_UAV)) - np.eye(N_UAV)).astype(int)

        # CBBA
        t0 = time.perf_counter()
        path_c, rnd_c = run_cbba(agents, tasks, world, compat, base_adj, p_loss,
                                  np.random.default_rng(run))
        cbba_t.append((time.perf_counter() - t0) * 1000.0)  # мс
        cbba_tcr.append(task_completion_rate(path_c, N_TASKS))
        cbba_rnd.append(rnd_c)
        cbba_cfr.append(conflict_free_rate(path_c))

        # A-CBBA
        t0 = time.perf_counter()
        path_a, rnd_a = run_acbba(agents, tasks, world, compat, base_adj, p_loss,
                                   np.random.default_rng(run+500))
        acbba_t.append((time.perf_counter() - t0) * 1000.0)
        acbba_tcr.append(task_completion_rate(path_a, N_TASKS))
        acbba_rnd.append(rnd_a)
        acbba_cfr.append(conflict_free_rate(path_a))

    raw['cbba']['tcr'].append(np.array(cbba_tcr))
    raw['cbba']['rounds'].append(np.array(cbba_rnd))
    raw['cbba']['cfr'].append(np.array(cbba_cfr))
    raw['cbba']['time_ms'].append(np.array(cbba_t))
    raw['acbba']['tcr'].append(np.array(acbba_tcr))
    raw['acbba']['rounds'].append(np.array(acbba_rnd))
    raw['acbba']['cfr'].append(np.array(acbba_cfr))
    raw['acbba']['time_ms'].append(np.array(acbba_t))

    print(f"  p_loss={p_loss:.0%}  CBBA TCR={np.mean(cbba_tcr):.3f}±{np.std(cbba_tcr, ddof=1):.3f}  "
          f"|  A-CBBA TCR={np.mean(acbba_tcr):.3f}±{np.std(acbba_tcr, ddof=1):.3f}  "
          f"|  t̄(CBBA)={np.mean(cbba_t):.1f}мс  t̄(A-CBBA)={np.mean(acbba_t):.1f}мс")

# Статистика: mean, median, std, IQR, 95% CI
def ci95(arr):
    n = len(arr)
    se = np.std(arr, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n-1)
    return t_crit * se

def iqr(arr):
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))

stats_summary = {alg: {m: {} for m in ('tcr', 'rounds', 'cfr')} for alg in ('cbba', 'acbba')}
for alg in ('cbba', 'acbba'):
    for m in ('tcr', 'rounds', 'cfr'):
        stats_summary[alg][m]['mean']   = [float(np.mean(x))        for x in raw[alg][m]]
        stats_summary[alg][m]['median'] = [float(np.median(x))      for x in raw[alg][m]]
        stats_summary[alg][m]['std']    = [float(np.std(x, ddof=1)) for x in raw[alg][m]]
        stats_summary[alg][m]['iqr']    = [iqr(x)                   for x in raw[alg][m]]
        stats_summary[alg][m]['ci']     = [float(ci95(x))           for x in raw[alg][m]]

# Парный t-тест + Уилкоксон + Cohen's d на каждом уровне p_loss
def cohens_d_paired(a, b):
    """Парный Cohen's d = mean(diff) / std(diff)."""
    diff = a - b
    return float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0

print("\n" + "="*78)
print("Парный t-тест + Уилкоксона + Cohen's d: A-CBBA vs CBBA на каждом уровне p_loss")
print(f"Коррекция Бонферрони: α = {ALPHA} / {N_COMPARISONS} = {ALPHA_BONF:.4f}")
print("="*78)
print(f"{'p_loss':>8} | {'ΔTCR п.п.':>10} | {'t-стат.':>9} | {'p(t)':>8} | "
      f"{'p(t)×Bonf':>10} | {'W':>8} | {'p(W)':>8} | {'d':>7} | {'знач.?':>7}")
print("-"*90)

ttest_rows = []
for i, p_loss in enumerate(P_LOSS_LEVELS):
    cbba_tcr  = raw['cbba']['tcr'][i]
    acbba_tcr = raw['acbba']['tcr'][i]
    diff      = acbba_tcr - cbba_tcr
    delta_pp  = float(np.mean(diff)) * 100

    # t-тест
    if np.allclose(diff, 0):
        t_stat, p_t = 0.0, 1.0
    else:
        t_stat, p_t = stats.ttest_rel(acbba_tcr, cbba_tcr)
    p_t_bonf = min(1.0, float(p_t) * N_COMPARISONS)

    # Критерий знаковых рангов Уилкоксона (парный, двусторонний)
    if np.allclose(diff, 0):
        w_stat, p_w = 0.0, 1.0
    else:
        w_stat, p_w = stats.wilcoxon(acbba_tcr, cbba_tcr, alternative='two-sided')

    # Cohen's d (парный)
    d = cohens_d_paired(acbba_tcr, cbba_tcr)

    # Значимо, если оба теста дают p < α_bonf
    significant = bool(p_t_bonf < ALPHA and float(p_w) < ALPHA_BONF)
    verdict = "ДА" if significant else "нет"

    print(f"{p_loss:>8.0%} | {delta_pp:>+10.2f} | {float(t_stat):>+9.3f} | {float(p_t):>8.4g} | "
          f"{p_t_bonf:>10.4g} | {float(w_stat):>8.1f} | {float(p_w):>8.4g} | {d:>+7.3f} | {verdict:>7}")

    ttest_rows.append({
        'p_loss_%':                    int(p_loss * 100),
        'delta_tcr_pp':                round(delta_pp, 3),
        't_stat':                      round(float(t_stat), 4),
        'p_value_ttest':               float(p_t),
        'p_value_ttest_bonferroni':    p_t_bonf,
        'W_wilcoxon':                  float(w_stat),
        'p_value_wilcoxon':            float(p_w),
        'cohens_d':                    round(d, 4),
        'significant_both_corrected':  significant,
    })

# Двухфакторный ANOVA с интеракцией
rows = []
for i, p_loss in enumerate(P_LOSS_LEVELS):
    for alg in ('cbba', 'acbba'):
        for v in raw[alg]['tcr'][i]:
            rows.append({'algorithm': alg, 'p_loss': float(p_loss), 'tcr': float(v)})
df = pd.DataFrame(rows)

model = ols('tcr ~ C(algorithm) * C(p_loss)', data=df).fit()
anova_table = anova_lm(model, typ=2)

print("\n" + "="*78)
print("Двухфакторный ANOVA: TCR ~ algorithm × p_loss")
print("="*78)
print(anova_table)

interaction_p = float(anova_table.loc['C(algorithm):C(p_loss)', 'PR(>F)'])
algo_p        = float(anova_table.loc['C(algorithm)', 'PR(>F)'])
ploss_p       = float(anova_table.loc['C(p_loss)', 'PR(>F)'])

print(f"\nГлавный эффект алгоритма:           p = {algo_p:.4g}  → "
      f"{'значим' if algo_p < ALPHA else 'не значим'}")
print(f"Главный эффект p_loss:              p = {ploss_p:.4g}  → "
      f"{'значим' if ploss_p < ALPHA else 'не значим'}")
print(f"Интеракция algorithm × p_loss:      p = {interaction_p:.4g}  → "
      f"{'значима — выигрыш A-CBBA зависит от p_loss' if interaction_p < ALPHA else 'не значима'}")

# Коэффициент Спирмена ρ: деградация TCR с ростом p_loss
print("\n" + "="*78)
print("Коэффициент Спирмена ρ(TCR, p_loss) — мера деградации каждого алгоритма")
print("="*78)

p_loss_arr = np.array(P_LOSS_LEVELS)
degradation_rows = []

for alg in ('cbba', 'acbba'):
    means_tcr = np.array(stats_summary[alg]['tcr']['mean'])
    rho, p_rho = stats.spearmanr(p_loss_arr, means_tcr)
    tcr_base   = means_tcr[0]                              # при p_loss = 0
    # Абсолютная деградация: TCR(p_loss) - TCR(0) в п.п.
    abs_deg    = (means_tcr - tcr_base) * 100              # отрицательные значения = деградация
    # Относительная: процент от базового уровня
    rel_deg    = (means_tcr / tcr_base - 1.0) * 100
    print(f"\n  {alg.upper()}: ρ = {rho:+.4f},  p = {p_rho:.4g}  "
          f"({'значимая' if p_rho < ALPHA else 'незначимая'} монотонная связь)")
    print(f"  {'p_loss':>8} | {'TCR mean%':>10} | {'Δabs п.п.':>10} | {'Δrel %':>10}")
    print("  " + "-"*46)
    for j, p in enumerate(P_LOSS_LEVELS):
        print(f"  {p:>8.0%} | {means_tcr[j]*100:>10.2f} | {abs_deg[j]:>+10.2f} | {rel_deg[j]:>+10.2f}")
        degradation_rows.append({
            'algorithm':    alg,
            'p_loss_%':     int(p * 100),
            'tcr_mean_%':   round(float(means_tcr[j]) * 100, 3),
            'delta_abs_pp': round(float(abs_deg[j]), 3),
            'delta_rel_%':  round(float(rel_deg[j]), 3),
            'spearman_rho': round(float(rho), 4),
            'spearman_p':   float(p_rho),
        })

# ── Сохранение в файлы ───────────────────────────────────────────
with open('results/summary_table.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow([
        'p_loss_%',
        'cbba_tcr_mean_%', 'cbba_tcr_median_%', 'cbba_tcr_std_%', 'cbba_tcr_iqr_%', 'cbba_tcr_ci95_%',
        'cbba_rounds_mean', 'cbba_rounds_median', 'cbba_rounds_std', 'cbba_rounds_iqr', 'cbba_rounds_ci95',
        'cbba_cfr_mean_%', 'cbba_cfr_median_%', 'cbba_cfr_std_%', 'cbba_cfr_iqr_%', 'cbba_cfr_ci95_%',
        'acbba_tcr_mean_%', 'acbba_tcr_median_%', 'acbba_tcr_std_%', 'acbba_tcr_iqr_%', 'acbba_tcr_ci95_%',
        'acbba_rounds_mean', 'acbba_rounds_median', 'acbba_rounds_std', 'acbba_rounds_iqr', 'acbba_rounds_ci95',
        'acbba_cfr_mean_%', 'acbba_cfr_median_%', 'acbba_cfr_std_%', 'acbba_cfr_iqr_%', 'acbba_cfr_ci95_%',
    ])
    for i, p in enumerate(P_LOSS_LEVELS):
        row_out = [int(p * 100)]
        for alg in ('cbba', 'acbba'):
            for m, scale in (('tcr', 100), ('rounds', 1), ('cfr', 100)):
                row_out.append(round(stats_summary[alg][m]['mean'][i]   * scale, 2))
                row_out.append(round(stats_summary[alg][m]['median'][i] * scale, 2))
                row_out.append(round(stats_summary[alg][m]['std'][i]    * scale, 2))
                row_out.append(round(stats_summary[alg][m]['iqr'][i]    * scale, 2))
                row_out.append(round(stats_summary[alg][m]['ci'][i]     * scale, 2))
        w.writerow(row_out)

with open('results/statistical_tests.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=ttest_rows[0].keys())
    w.writeheader()
    w.writerows(ttest_rows)

with open('results/degradation.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=degradation_rows[0].keys())
    w.writeheader()
    w.writerows(degradation_rows)

with open('results/anova_results.txt', 'w') as f:
    f.write("Двухфакторный ANOVA: TCR ~ algorithm × p_loss\n")
    f.write(f"N = {N_RUNS} прогонов на ячейку, всего наблюдений: {len(df)}\n")
    f.write("="*78 + "\n\n")
    f.write(str(anova_table) + "\n\n")
    f.write(f"Главный эффект алгоритма:      p = {algo_p:.6g}\n")
    f.write(f"Главный эффект p_loss:         p = {ploss_p:.6g}\n")
    f.write(f"Интеракция algorithm × p_loss: p = {interaction_p:.6g}\n")
    f.write(f"\nЗаключение: интеракция {'значима' if interaction_p < ALPHA else 'не значима'} "
            f"при α = {ALPHA}.\n")
    if interaction_p < ALPHA:
        f.write("Это подтверждает гипотезу, что выигрыш A-CBBA "
                "нелинейно растёт с увеличением вероятности потерь пакетов.\n")

with open('results/raw_runs.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['p_loss_%', 'algorithm', 'run_idx', 'tcr', 'rounds', 'cfr', 'time_ms'])
    for i, p in enumerate(P_LOSS_LEVELS):
        for alg in ('cbba', 'acbba'):
            for k in range(N_RUNS):
                w.writerow([
                    int(p*100), alg, k,
                    round(float(raw[alg]['tcr'][i][k]), 4),
                    int(raw[alg]['rounds'][i][k]),
                    round(float(raw[alg]['cfr'][i][k]), 4),
                    round(float(raw[alg]['time_ms'][i][k]), 3),
                ])

# Построение графиков
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

# TCR с CI
fig, ax = make_fig()
for alg in ('cbba', 'acbba'):
    means = np.array(stats_summary[alg]['tcr']['mean']) * 100
    cis   = np.array(stats_summary[alg]['tcr']['ci'])   * 100
    ax.errorbar(xs, means, yerr=cis, capsize=3, **STYLE[alg])
ax.set_ylabel('Доля выполненных задач (%)', fontsize=10)
ax.set_title('TCR vs потери пакетов (95% CI по 50 прогонам)', fontsize=10)
ax.set_ylim(70, 100)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig('results/tcr_vs_ploss.png', dpi=150)
plt.close(fig)

# Раунды с CI
fig, ax = make_fig()
for alg in ('cbba', 'acbba'):
    means = np.array(stats_summary[alg]['rounds']['mean'])
    cis   = np.array(stats_summary[alg]['rounds']['ci'])
    ax.errorbar(xs, means, yerr=cis, capsize=3, **STYLE[alg])
ax.set_ylabel('Раунды консенсуса (средн.)', fontsize=10)
ax.set_title('Раунды консенсуса vs потери пакетов (95% CI)', fontsize=10)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig('results/rounds_vs_ploss.png', dpi=150)
plt.close(fig)

# CFR с CI
fig, ax = make_fig()
for alg in ('cbba', 'acbba'):
    means = np.array(stats_summary[alg]['cfr']['mean']) * 100
    cis   = np.array(stats_summary[alg]['cfr']['ci'])   * 100
    ax.errorbar(xs, means, yerr=cis, capsize=3, **STYLE[alg])
ax.set_ylabel('Доля бесконфликтных назначений (%)', fontsize=10)
ax.set_title('CFR vs потери пакетов (95% CI)', fontsize=10)
ax.set_ylim(0, 100)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig('results/cfr_vs_ploss.png', dpi=150)
plt.close(fig)

# Комбинированный
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, key, scale, ylabel, title, ylim in zip(
    axes, ['tcr', 'rounds'], [100, 1],
    ['Доля выполненных задач (%)', 'Раунды консенсуса (средн.)'],
    ['(a) Доля выполненных задач', '(b) Раунды консенсуса'],
    [(70,100), None],
):
    for alg in ('cbba', 'acbba'):
        means = np.array(stats_summary[alg][key]['mean']) * scale
        cis   = np.array(stats_summary[alg][key]['ci'])   * scale
        ax.errorbar(xs, means, yerr=cis, capsize=3, **STYLE[alg])
    ax.set_xlabel('Потери пакетов (%)', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
    if ylim:
        ax.set_ylim(*ylim)
fig.tight_layout()
fig.savefig('results/combined_figure.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for alg in ('cbba', 'acbba'):
    tcr_means = np.array(stats_summary[alg]['tcr']['mean']) * 100
    tcr_cis   = np.array(stats_summary[alg]['tcr']['ci'])   * 100
    axes[0].errorbar(xs, tcr_means, yerr=tcr_cis, capsize=3, **STYLE[alg])
    time_means = np.array([np.mean(t) for t in raw[alg]['time_ms']])
    time_cis   = np.array([ci95(t)    for t in raw[alg]['time_ms']])
    axes[1].errorbar(xs, time_means, yerr=time_cis, capsize=3, **STYLE[alg])
axes[0].set_xlabel('Потери пакетов (%)', fontsize=10)
axes[0].set_ylabel('Доля выполненных задач (%)', fontsize=10)
axes[0].set_title('(a) Доля выполненных задач', fontsize=10)
axes[0].set_ylim(70, 100)
axes[0].grid(True, linestyle=':', alpha=0.5)
axes[0].legend(fontsize=9)
axes[0].tick_params(labelsize=9)
axes[1].set_xlabel('Потери пакетов (%)', fontsize=10)
axes[1].set_ylabel('Время одного вызова solve(), мс', fontsize=10)
axes[1].set_title('(b) Время выполнения', fontsize=10)
axes[1].grid(True, linestyle=':', alpha=0.5)
axes[1].legend(fontsize=9)
axes[1].tick_params(labelsize=9)
fig.tight_layout()
fig.savefig('results/tcr_time_figure.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print("\n" + "="*78)
print("Готово. Результаты сохранены в папку results/:")
print("  summary_table.csv      — mean, median, std, IQR, 95% CI для всех метрик")
print("  statistical_tests.csv  — t-тест + Уилкоксон + Cohen's d + Бонферрони")
print("  anova_results.txt      — двухфакторный ANOVA с интеракцией")
print("  degradation.csv        — Spearman ρ и деградация TCR по уровням p_loss")
print("  raw_runs.csv           — все прогоны с time_ms")
print("  performance.txt        — замеры времени и потребляемых ресурсов")
print("  *.png                  — графики с error bars")
print("="*78)

# Замеры времени выполнения и используемых ресурсов
T_total = time.perf_counter() - T_total_start
T_cpu   = time.process_time() - T_cpu_start
peak_mem_python_mb = tracemalloc.get_traced_memory()[1] / 1024**2
tracemalloc.stop()

all_cbba_times  = np.concatenate(raw['cbba']['time_ms'])
all_acbba_times = np.concatenate(raw['acbba']['time_ms'])
total_runs = len(all_cbba_times) + len(all_acbba_times)

cpu_load_pct = 100.0 * T_cpu / T_total if T_total > 0 else 0.0

rss_mb = proc.memory_info().rss / 1024**2

# Коэффициент накладных расходов A-CBBA/CBBA по каждому уровню
perf_lines = []
perf_lines.append("="*78)
perf_lines.append("Замеры времени и использованных ресурсов")
perf_lines.append("="*78)
perf_lines.append(f"  Дата запуска:                 {time.strftime('%Y-%m-%d %H:%M:%S')}")
perf_lines.append(f"  Общее wall-clock время:       {T_total:.2f} с ({T_total/60:.2f} мин)")
perf_lines.append(f"  CPU-время процесса:           {T_cpu:.2f} с")
perf_lines.append(f"  Загрузка CPU (CPU/wall):      {cpu_load_pct:.1f}% "
                  f"(≈ {cpu_load_pct/100:.2f} ядра)")
perf_lines.append(f"  Всего прогонов:               {total_runs} (по {N_RUNS} на каждое из "
                  f"{2*len(P_LOSS_LEVELS)} условий)")
perf_lines.append("")
perf_lines.append("  Сводная статистика времени одного прогона (мс, по всем уровням p_loss):")
for label, arr in (("CBBA   ", all_cbba_times), ("A-CBBA ", all_acbba_times)):
    perf_lines.append(
        f"    {label} — среднее: {np.mean(arr):7.2f} ± {np.std(arr, ddof=1):6.2f},  "
        f"медиана: {np.median(arr):7.2f},  p95: {np.percentile(arr, 95):7.2f},  "
        f"max: {np.max(arr):7.2f}"
    )
overhead_mean   = np.mean(all_acbba_times) / np.mean(all_cbba_times)
overhead_median = np.median(all_acbba_times) / np.median(all_cbba_times)
perf_lines.append(f"    Коэффициент накладных расходов A-CBBA/CBBA:  "
                  f"по среднему = {overhead_mean:.3f}×  ({(overhead_mean-1)*100:+.1f}%),  "
                  f"по медиане = {overhead_median:.3f}×  ({(overhead_median-1)*100:+.1f}%)")
perf_lines.append("")
perf_lines.append("  Время по уровню p_loss (мс), среднее ± std  |  коэф. нагрузки:")
perf_lines.append(f"  {'p_loss':>8} | {'CBBA':>18} | {'A-CBBA':>18} | {'overhead':>10}")
perf_lines.append("  " + "-"*62)
for i, p_loss in enumerate(P_LOSS_LEVELS):
    c = raw['cbba']['time_ms'][i]
    a = raw['acbba']['time_ms'][i]
    ov = np.mean(a) / np.mean(c) if np.mean(c) > 0 else float('nan')
    perf_lines.append(
        f"  {p_loss:>8.0%} | "
        f"{np.mean(c):7.2f} ± {np.std(c, ddof=1):5.2f}    | "
        f"{np.mean(a):7.2f} ± {np.std(a, ddof=1):5.2f}    | "
        f"{ov:>10.3f}×"
    )
perf_lines.append("")
perf_lines.append(f"  Пиковое потребление памяти Python (tracemalloc): {peak_mem_python_mb:.2f} МБ")
if rss_mb is not None:
    perf_lines.append(f"  Размер процесса в RAM (RSS, psutil):             {rss_mb:.2f} МБ")
perf_lines.append("="*78)

perf_text = "\n".join(perf_lines)
print("\n" + perf_text)

with open('results/performance.txt', 'w') as f:
    f.write("Окружение:\n")
    f.write(f"  ОС:               {platform.system()} {platform.release()}\n")
    f.write(f"  Архитектура:      {platform.machine()}\n")
    f.write(f"  Процессор:        {platform.processor() or 'не определён'}\n")
    f.write(f"  Логических ядер:  {os.cpu_count()}\n")
    f.write(f"  Физических ядер:  {psutil.cpu_count(logical=False)}\n")
    f.write(f"  ОЗУ всего:        {psutil.virtual_memory().total/1024**3:.1f} ГБ\n")
    f.write(f"  Python:           {platform.python_version()}\n")
    f.write(f"  NumPy:            {np.__version__}\n")
    import scipy as _sp
    f.write(f"  SciPy:            {_sp.__version__}\n")
    import statsmodels as _sm
    f.write(f"  statsmodels:      {_sm.__version__}\n\n")
    f.write(perf_text + "\n")
