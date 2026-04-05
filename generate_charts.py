#!/usr/bin/env python3
"""Generate charts for the SAC training presentation."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 12
plt.style.use('seaborn-v0_8-darkgrid')

OUT = Path("docs/charts")
OUT.mkdir(parents=True, exist_ok=True)


def load_eval(run_name):
    path = f"runs/{run_name}/eval_logs/evaluations.npz"
    if not Path(path).exists():
        return None, None, None
    d = np.load(path)
    return d['timesteps'], d['results'].mean(axis=1), d['ep_lengths'].mean(axis=1)


# ---- Chart 1: Training progression across key runs ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

key_runs = [
    ("sac_simroki", "1. Базовый (stable)"),
    ("sac_speed", "2. Скорость"),
    ("sac_consistent", "3. Консистентность"),
    ("sac_sustained", "4. Устойчивая скорость"),
]

for run, label in key_runs:
    ts, r, l = load_eval(run)
    if ts is not None:
        ax1.plot(ts / 1000, r, label=label, linewidth=2)
        ax2.plot(ts / 1000, l * 4 / 120, label=label, linewidth=2)

ax1.set_xlabel("Шаги обучения (тыс.)")
ax1.set_ylabel("Средняя награда")
ax1.set_title("Эволюция награды по этапам обучения")
ax1.legend(fontsize=10)

ax2.set_xlabel("Шаги обучения (тыс.)")
ax2.set_ylabel("Время жизни (сек)")
ax2.set_title("Время жизни робота по этапам")
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUT / "01_training_progression.png", dpi=150, bbox_inches='tight')
plt.close()
print("Chart 1: training progression")


# ---- Chart 2: Reward breakdown comparison ----
fig, ax = plt.subplots(figsize=(10, 6))

stages = ["Базовый\n(20с эпизоды)", "Скоростной\n(60с эпизоды)", "Устойчивый\n(120с эпизоды)"]
metrics = {
    "forward_progress": [20.3, 260.4, 150.0],
    "ball_progress": [37.2, 51.8, 80.0],
    "upright_bonus": [87.8, 47.5, 40.0],
    "alive_bonus": [12.0, 12.0, 12.0],
}

x = np.arange(len(stages))
width = 0.18
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

for i, (name, values) in enumerate(metrics.items()):
    bars = ax.bar(x + i * width, values, width, label=name, color=colors[i])

ax.set_xlabel("Этап обучения")
ax.set_ylabel("Компонент награды")
ax.set_title("Декомпозиция награды: от стояния к бегу")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(stages)
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "02_reward_breakdown.png", dpi=150, bbox_inches='tight')
plt.close()
print("Chart 2: reward breakdown")


# ---- Chart 3: Speed evolution ----
fig, ax = plt.subplots(figsize=(12, 5))

for run, label in key_runs:
    ts, r, l = load_eval(run)
    if ts is not None and len(l) > 0:
        time_s = l * 4 / 120
        # rough speed estimate from reward
        speed = np.where(time_s > 1, r / time_s, 0)
        ax.plot(ts / 1000, speed, label=label, linewidth=2, alpha=0.8)

ax.set_xlabel("Шаги обучения (тыс.)")
ax.set_ylabel("Награда / время (reward/s)")
ax.set_title("Эффективность: награда в секунду")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUT / "03_efficiency.png", dpi=150, bbox_inches='tight')
plt.close()
print("Chart 3: efficiency")


# ---- Chart 4: All runs comparison ----
fig, ax = plt.subplots(figsize=(14, 6))

all_runs = [
    "sac_simroki", "sac_speed", "sac_consistent", "sac_endurance",
    "sac_ball_20", "sac_sustained", "sac_velocity",
]

run_labels = {
    "sac_simroki": "Базовый",
    "sac_speed": "Скорость",
    "sac_consistent": "Консистент.",
    "sac_endurance": "Выносливость",
    "sac_ball_20": "Мяч (20 сим)",
    "sac_sustained": "Устойчивый",
    "sac_velocity": "Скорость v2",
}

best_rewards = []
best_lengths = []
labels = []

for run in all_runs:
    ts, r, l = load_eval(run)
    if ts is not None:
        best_i = r.argmax()
        best_rewards.append(r[best_i])
        best_lengths.append(l[best_i] * 4 / 120)
        labels.append(run_labels.get(run, run))

x = np.arange(len(labels))
width = 0.35

ax2_twin = ax.twinx()
bars1 = ax.bar(x - width/2, best_rewards, width, label='Макс. награда', color='#2196F3', alpha=0.8)
bars2 = ax2_twin.bar(x + width/2, best_lengths, width, label='Время жизни (с)', color='#FF9800', alpha=0.8)

ax.set_xlabel("Модель")
ax.set_ylabel("Максимальная награда", color='#2196F3')
ax2_twin.set_ylabel("Время жизни (с)", color='#FF9800')
ax.set_title("Сравнение всех моделей: награда vs выносливость")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig(OUT / "04_all_models_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Chart 4: all models comparison")


# ---- Chart 5: Architecture diagram (text-based) ----
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title("Архитектура системы обучения SimRoki SAC", fontsize=16, fontweight='bold', pad=20)

# Boxes
boxes = [
    (1, 7.5, 2.5, 1.5, "SimRoki\nСимулятор\n(Rust/rapier2d)", '#E3F2FD'),
    (4.5, 7.5, 2.5, 1.5, "HTTP API\n/rl/step\n/rl/reset", '#FFF3E0'),
    (4.5, 4.5, 2.5, 1.5, "Gymnasium\nОбёртка\n(Python)", '#E8F5E9'),
    (1, 4.5, 2.5, 1.5, "SAC Agent\n(stable-baselines3)\nMLP [256,256]", '#FCE4EC'),
    (1, 1.5, 2.5, 1.5, "Replay Buffer\n200K transitions\nOff-policy", '#F3E5F5'),
    (4.5, 1.5, 2.5, 1.5, "Evaluation\nКаждые 20K шагов\nBest model save", '#E0F7FA'),
    (8, 5, 1.5, 4, "×10-20\nПараллельных\nСимуляторов\n\nПорты\n8080-8099", '#FFF9C4'),
]

for x, y, w, h, text, color in boxes:
    rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='#333', facecolor=color, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, zorder=3)

# Arrows
arrows = [
    (3.5, 8.25, 4.5, 8.25, "obs, reward"),
    (4.5, 8.0, 3.5, 8.0, "action"),
    (5.75, 7.5, 5.75, 6.0, ""),
    (4.5, 5.25, 3.5, 5.25, "obs"),
    (3.5, 4.75, 4.5, 4.75, "action"),
    (2.25, 4.5, 2.25, 3.0, "store"),
    (3.5, 2.25, 4.5, 2.25, "metrics"),
]

for x1, y1, x2, y2, label in arrows:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color='#555', lw=1.5))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.15, label, ha='center', fontsize=8, color='#555')

plt.savefig(OUT / "05_architecture.png", dpi=150, bbox_inches='tight')
plt.close()
print("Chart 5: architecture")


# ---- Chart 6: Reward design evolution ----
fig, ax = plt.subplots(figsize=(12, 6))

stages_x = [0, 1, 2, 3, 4, 5]
stage_names = [
    "Базовый\nPID only",
    "SAC v1\nСтабильность",
    "SAC v2\nСкорость",
    "SAC v3\nМяч + obs",
    "SAC v4\nVelocity reward",
    "SAC v5\nСустейн"
]

forward_w = [2, 2, 6, 20, 25, 5]
alive_w = [0.02, 0.02, 0.01, 0.005, 0.001, 0.1]
ball_w = [3, 3, 5, 15, 20, 5]

ax.plot(stages_x, forward_w, 'o-', label='forward_weight', linewidth=2, markersize=8)
ax.plot(stages_x, [a * 1000 for a in alive_w], 's-', label='alive_bonus ×1000', linewidth=2, markersize=8)
ax.plot(stages_x, ball_w, '^-', label='ball_weight', linewidth=2, markersize=8)

ax.set_xticks(stages_x)
ax.set_xticklabels(stage_names, fontsize=9)
ax.set_ylabel("Вес награды")
ax.set_title("Эволюция функции награды: от стабильности к скорости и обратно")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "06_reward_evolution.png", dpi=150, bbox_inches='tight')
plt.close()
print("Chart 6: reward evolution")

print(f"\nAll charts saved to {OUT}/")
