import numpy as np
import matplotlib.pyplot as plt
from requid_dem_model import GPUAgent

def plot_ability_distribution(n_agents=1000):
    # エージェントの生成
    agent = GPUAgent(n_agents, device='cpu')
    
    # 能力値の取得
    abilities = agent.abilities
    
    # プロットの設定
    plt.figure(figsize=(10, 6))
    
    # ヒストグラム
    plt.hist(abilities, bins=50, density=True, alpha=0.7, color='blue', label='実際の分布')
    
    # 理論的なベータ分布（α=2, β=3）
    x = np.linspace(0, 1, 100)
    from scipy.stats import beta
    plt.plot(x, beta.pdf(x, 2, 3), 'r-', lw=2, label='理論分布 (α=2, β=3)')
    
    # グラフの装飾
    plt.title('エージェントの能力分布')
    plt.xlabel('能力値')
    plt.ylabel('密度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 統計情報の表示
    mean_ability = np.mean(abilities)
    median_ability = np.median(abilities)
    std_ability = np.std(abilities)
    
    stats_text = f'平均: {mean_ability:.3f}\n中央値: {median_ability:.3f}\n標準偏差: {std_ability:.3f}'
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存
    plt.savefig('ability_distribution.png')
    plt.close()

if __name__ == '__main__':
    plot_ability_distribution() 