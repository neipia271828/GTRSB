"""
Shogi population dynamics simulator
===================================
Requires:
  pip install numpy pandas matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# PARAMETERS – モデル設定
# ---------------------------------------------------------------------------
TOTAL_POP0          = 120_000_000      # 人
BIRTHS_PER_YEAR     = 750_000          # 人/年
LIFESPAN            = 80               # 歳
STAGE_BOUNDS        = [(0, 20),        # 若年層
                       (20, 60),       # 中年層
                       (60, 80)]       # 老年層
ATTEMPT_PER_STAGE   = 1                # 勧誘回数/ステージ
RECRUIT_PROB_STAGE  = [0.5, 0.25, 0.5] # 勧誘成功確率
INIT_COMP_STAGE_FRAC = [0.4, 0.2, 0.4] # 初期層別競技者比率

# ---------------------------------------------------------------------------
# 事前計算
# ---------------------------------------------------------------------------
STAGE_LENGTH = [hi - lo for lo, hi in STAGE_BOUNDS]
P_EFFECTIVE  = [RECRUIT_PROB_STAGE[i] * ATTEMPT_PER_STAGE / STAGE_LENGTH[i]
                for i in range(3)]  # 年平均勧誘成功率

# ---------------------------------------------------------------------------
# シミュレーション関数
# ---------------------------------------------------------------------------
def simulate(initial_competitors: int, years: int):
    """
    初期競技人口 `initial_competitors` で `years` 年進めた後の競技人口を返す。
    """
    competitors = initial_competitors
    for _ in range(years):
        # 各年齢層の勧誘をシミュレート
        for i in range(3):
            # 勧誘回数を計算
            attempts = int(competitors * ATTEMPT_PER_STAGE / STAGE_LENGTH[i])
            # 勧誘成功確率を計算（100%以上の場合、1人確定 + 残り確率で追加）
            if P_EFFECTIVE[i] >= 1.0:
                successes = attempts + int((P_EFFECTIVE[i] - 1.0) * attempts)
            else:
                successes = int(attempts * P_EFFECTIVE[i])
            competitors += successes
        # 寿命による減少をシミュレート
        competitors = int(competitors * (1 - 1 / LIFESPAN))
    return competitors

def trajectory(initial_comp: int, years: int = 300):
    """
    年ごとの競技人口推移を np.array で返す。
    """
    ages = np.full(LIFESPAN, TOTAL_POP0 // LIFESPAN, dtype=float)
    comp = np.zeros(LIFESPAN, dtype=float)
    for i_stage, (lo, hi) in enumerate(STAGE_BOUNDS):
        comp[lo:hi] = initial_comp * INIT_COMP_STAGE_FRAC[i_stage] / STAGE_LENGTH[i_stage]
    noncomp = ages - comp

    totals = np.empty(years + 1)
    totals[0] = comp.sum()

    for t in range(1, years + 1):
        # 勧誘
        for i_stage, (lo, hi) in enumerate(STAGE_BOUNDS):
            comp_s    = comp[lo:hi].sum()
            noncomp_s = noncomp[lo:hi].sum()
            recruits  = min(noncomp_s, comp_s * P_EFFECTIVE[i_stage])
            if recruits:
                ratio   = recruits / noncomp_s
                gained  = noncomp[lo:hi] * ratio
                comp[lo:hi]    += gained
                noncomp[lo:hi] -= gained

        # 年齢進行
        comp[1:], comp[0]       = comp[:-1], 0
        noncomp[1:], noncomp[0] = noncomp[:-1], BIRTHS_PER_YEAR
        comp[-1] = noncomp[-1] = 0
        totals[t] = comp.sum()

    return totals

# ---------------------------------------------------------------------------
# 可視化ユーティリティ
# ---------------------------------------------------------------------------
def plot_trajectories(initial_list, years=300):
    """
    初期競技人口のリストをまとめて描画。
    """
    plt.figure(figsize=(9, 5))
    for init in initial_list:
        y = trajectory(init, years)
        plt.plot(y, label=f"{init/1e6:.1f} M start")
    plt.title("Shogi competitor trajectories")
    plt.xlabel("Year")
    plt.ylabel("Competitors")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# 閾値探索（任意）
# ---------------------------------------------------------------------------
def find_threshold(scan_min=1_000_000, scan_max=10_000_000, step=500_000, years=200):
    vals = np.arange(scan_min, scan_max + step, step)
    growth = [simulate(v, years) - v for v in vals]
    df = pd.DataFrame({"Initial": vals, "Growth": growth})
    # サインが変わる境界を探す
    try:
        idx = np.where(np.sign(growth[:-1]) != np.sign(growth[1:]))[0][0]
        threshold = vals[idx + 1]
    except IndexError:
        threshold = None
    return df, threshold

def equilibrium_gd(start=6_000_000,
                   years=200,
                   lr=1.0,
                   delta=50_000,
                   tol=1e-1,
                   max_iter=50,
                   verbose=True):
    """
    Gradient-descent style root-finder for f(x) = simulate(x) − x.
    Returns the initial competitor population x* that makes f(x*) ≈ 0.

    Parameters
    ----------
    start : int
        Starting guess for initial competitors.
    years : int
        Horizon passed to simulate().
    lr : float
        Learning-rate multiplier (α). Typical 0.5–1.5。
    delta : int
        Finite-difference step for ∂f/∂x ≈ [f(x+δ)−f(x−δ)]/(2δ)
    tol : float
        |f(x)| < tol で収束判定（人ベース）。
    max_iter : int
        最大ステップ数。
    verbose : bool
        True なら各イテレーションを print。

    Notes
    -----
    - f(x) = simulate(x, years) − x  …「何人増減するか」  
    - 勾配 g = ∂f/∂x を有限差分で求め、x ← x − lr · f/g で更新  
      （Newton 法と同形だが "降下勾配" 表現）。
    """
    x = float(start)

    for i in range(max_iter):
        fx = simulate(int(x), years) - x
        if abs(fx) < tol:
            if verbose:
                print(f"[{i:02}] |f| < tol → 収束: {x:,.0f}")
            return x

        # 有限差分で勾配を近似
        fp = simulate(int(x + delta), years) - (x + delta)
        fm = simulate(int(x - delta), years) - (x - delta)
        g  = (fp - fm) / (2 * delta)

        # 勾配が極端に小さい場合は学習率を半減
        if abs(g) < 1e-6:
            lr *= 0.5
            if verbose:
                print(f"[{i:02}] |g| too small, lr→{lr}")
            continue

        # パラメータ更新
        x_new = x - lr * fx / g
        if verbose:
            print(f"[{i:02}] x={x:,.0f}, f={fx:,.1f}, g={g:.4f} ⇒ x_new={x_new:,.0f}")
        x = max(0, x_new)            # 負値にならないよう制限

    if verbose:
        print("Warning: max_iter reached without convergence.")
    return x

def optimize_probabilities(target_pop=100_000,
                         years=300,
                         lr=0.001,
                         delta=0.0001,
                         tol=1e-2,
                         max_iter=200,
                         verbose=True):
    """
    各年齢層の勧誘成功確率を最適化する勾配降下法。
    
    Parameters
    ----------
    target_pop : int
        目標とする競技人口（デフォルト: 10万人）。
    years : int
        シミュレーション期間（デフォルト: 300年）。
    lr : float
        学習率（0.001程度が推奨）。
    delta : float
        確率の差分ステップ（0.0001程度が推奨）。
    tol : float
        収束判定の閾値（目標人口との差の割合）。
    max_iter : int
        最大イテレーション数。
    verbose : bool
        進捗表示の有無。
    
    Returns
    -------
    tuple
        (最適化された確率リスト, 最終的な競技人口)
    """
    global RECRUIT_PROB_STAGE, P_EFFECTIVE
    
    # 初期確率をコピー（より現実的な値から開始）
    probs = np.array([0.5, 0.25, 0.5], dtype=float)
    
    initial_competitors = 6_000_000  # 初期競技人口を600万人に固定
    
    for i in range(max_iter):
        # 現在の確率での競技人口を計算
        current_pop = simulate(initial_competitors, years)
        
        # 目標との差を計算
        error = (current_pop - target_pop) / target_pop
        
        if abs(error) < tol:
            if verbose:
                print(f"[{i:02}] 収束: 誤差={error:.1%}")
            break
        
        # 各確率に対する勾配を計算
        gradients = np.zeros_like(probs)
        for j in range(len(probs)):
            # 確率を少し増やす
            probs_plus = probs.copy()
            probs_plus[j] += delta
            probs_plus[j] = min(1.0, probs_plus[j])
            
            # 確率を少し減らす
            probs_minus = probs.copy()
            probs_minus[j] -= delta
            probs_minus[j] = max(0.0, probs_minus[j])
            
            # 差分を計算
            RECRUIT_PROB_STAGE = probs_plus
            P_EFFECTIVE = [RECRUIT_PROB_STAGE[i] * ATTEMPT_PER_STAGE / STAGE_LENGTH[i]
                          for i in range(3)]
            pop_plus = simulate(initial_competitors, years)
            
            RECRUIT_PROB_STAGE = probs_minus
            P_EFFECTIVE = [RECRUIT_PROB_STAGE[i] * ATTEMPT_PER_STAGE / STAGE_LENGTH[i]
                          for i in range(3)]
            pop_minus = simulate(initial_competitors, years)
            
            # 勾配を計算
            gradients[j] = (pop_plus - pop_minus) / (2 * delta)
        
        # 勾配の正規化
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > 0:
            gradients = gradients / grad_norm
        
        # 確率を更新（より慎重に）
        probs -= lr * error * gradients
        
        # 確率の範囲を制限（100%以上も許容）
        probs = np.clip(probs, 0.0, 1.0)
        
        # 確率を反映
        RECRUIT_PROB_STAGE = probs.tolist()
        P_EFFECTIVE = [RECRUIT_PROB_STAGE[i] * ATTEMPT_PER_STAGE / STAGE_LENGTH[i]
                      for i in range(3)]
        
        if verbose:
            print(f"[{i:02}] 確率={probs}, 人口={current_pop:,.0f}, 誤差={error:.1%}")
    
    return probs.tolist(), current_pop

# ---------------------------------------------------------------------------
# MAIN（スクリプトとして呼ばれた時のみ走る）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    YEARS = 300

    # 例1：600 万人、900 万人、1200 万人スタートを比較
    plot_trajectories([6_000_000, 9_000_000, 12_000_000], years=YEARS)

    # 例2：クリティカルマス探索
    df, th = find_threshold()
    print(df)
    print("Critical mass ≈", f"{th:_}" if th else "境界見つからず")

    # 目標人口1000万人で確率を最適化
    probs, pop = optimize_probabilities(target_pop=10_000_000, years=200, lr=0.001, delta=0.0001, verbose=True)
    print(f"最適化された確率: {probs}")
    print(f"最終的な競技人口: {pop:,.0f}")