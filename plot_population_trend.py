import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import importlib
import shogi_model

x = np.arange(301)  # 0年目〜300年目

# 降下勾配法のパラメータ
max_iter = 10000
lr = 0.0005  # 学習率を小さくして安定性を向上
probs = np.array([0.5, 0.3, 0.5])  # より高い初期確率から開始

tol_slope = 1e-1  # 傾き許容値を緩和
found_equilibrium = False

for i in range(max_iter):
    # 確率をshogi_modelに反映
    shogi_model.RECRUIT_PROB_STAGE = probs.tolist()
    shogi_model.P_EFFECTIVE = [shogi_model.RECRUIT_PROB_STAGE[j] * shogi_model.ATTEMPT_PER_STAGE / shogi_model.STAGE_LENGTH[j] for j in range(3)]
    # 300年分の人口推移
    pop_traj = shogi_model.trajectory(6_000_000, years=300)
    # 線形フィット
    popt, _ = curve_fit(lambda x, a, b: a * x + b, x, pop_traj)
    slope = popt[0]
    
    # 進捗表示（100回ごと）
    if i % 100 == 0:
        print(f'試行 {i}: 確率={probs}, 傾き={slope:.6f}')
    
    # 平衡状態ならグラフ描画
    if abs(slope) < tol_slope:
        plt.figure(figsize=(10, 6))
        plt.plot(x, pop_traj, label='競技人口')
        plt.plot(x, popt[0]*x + popt[1], 'r-', label=f'フィット: y = {popt[0]:.2f}x + {popt[1]:.2f}')
        plt.xlabel('年')
        plt.ylabel('競技人口')
        plt.title(f'平衡状態の競技人口の推移（試行{i+1}回目）')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'population_trend_equilibrium_{i+1}.png')
        print(f'平衡状態に到達: 試行{i+1}回目, 確率={probs}, 最適な線: y = {popt[0]:.2f}x + {popt[1]:.2f}')
        found_equilibrium = True
        break
    
    # 勾配計算（数値微分）
    grad = np.zeros_like(probs)
    delta = 1e-4
    for j in range(len(probs)):
        p_up = probs.copy()
        p_up[j] += delta
        shogi_model.RECRUIT_PROB_STAGE = p_up.tolist()
        shogi_model.P_EFFECTIVE = [shogi_model.RECRUIT_PROB_STAGE[k] * shogi_model.ATTEMPT_PER_STAGE / shogi_model.STAGE_LENGTH[k] for k in range(3)]
        pop_traj_up = shogi_model.trajectory(6_000_000, years=300)
        popt_up, _ = curve_fit(lambda x, a, b: a * x + b, x, pop_traj_up)
        grad[j] = (popt_up[0] - slope) / delta
    
    # パラメータ更新（より緩やかに）
    probs -= lr * grad
    # 確率の範囲を拡大（0.0〜2.0）
    probs = np.clip(probs, 0.0, 2.0)

if not found_equilibrium:
    print('平衡状態には到達しませんでした。')
    # 最後の状態をプロット
    plt.figure(figsize=(10, 6))
    plt.plot(x, pop_traj, label='競技人口')
    plt.plot(x, popt[0]*x + popt[1], 'r-', label=f'フィット: y = {popt[0]:.2f}x + {popt[1]:.2f}')
    plt.xlabel('年')
    plt.ylabel('競技人口')
    plt.title(f'最終状態の競技人口の推移（試行{max_iter}回目）')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('population_trend_final.png')
    print(f'最終状態: 確率={probs}, 傾き={slope:.6f}') 