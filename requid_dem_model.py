#!/usr/bin/env python3
"""
GPU加速版 液体民主主義シミュレーション
CuPy + PyTorchを使用した大規模エージェントベースモデル
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path

# GPU加速ライブラリ
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    GPU_AVAILABLE = True
    print("GPU (CuPy) available")
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False
    print("GPU not available, using CPU")

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        print(f"PyTorch GPU available: {torch.cuda.get_device_name()}")
except ImportError:
    TORCH_AVAILABLE = False

class GPUAgent:
    """GPU最適化エージェントクラス（バッチ処理用）"""
    
    def __init__(self, n_agents: int, device='gpu'):
        self.n_agents = n_agents
        self.device = device
        
        if device == 'gpu' and GPU_AVAILABLE:
            self.xp = cp
        else:
            self.xp = np
        
        # エージェント属性をバッチで初期化
        self.abilities = self.xp.random.beta(2, 3, size=n_agents).astype(self.xp.float32)
        self.ideologies = self.xp.random.normal(0, 1, size=n_agents).astype(self.xp.float32)
        self.agent_ids = self.xp.arange(n_agents, dtype=self.xp.int32)
        
        # 信頼行列 (sparse matrix for efficiency)
        if device == 'gpu' and GPU_AVAILABLE:
            self.trust_matrix = cp_sparse.csr_matrix((n_agents, n_agents), dtype=self.xp.float32)
        else:
            from scipy.sparse import csr_matrix
            self.trust_matrix = csr_matrix((n_agents, n_agents), dtype=np.float32)
    
    def batch_signal_generation(self, true_answers: 'xp.ndarray') -> 'xp.ndarray':
        """バッチでシグナル生成"""
        n_issues = len(true_answers)
        # (n_agents, n_issues) の形状
        random_vals = self.xp.random.rand(self.n_agents, n_issues)
        # 各エージェントの能力に基づいて正解を取得
        correct_mask = random_vals < self.abilities[:, None]
        
        # ブロードキャストを使って効率的に計算
        signals = self.xp.where(correct_mask, 
                               true_answers[None, :], 
                               1 - true_answers[None, :])
        return signals.astype(self.xp.int32)

class GPUNetworkGenerator:
    """GPU最適化ネットワーク生成"""
    
    @staticmethod
    def gpu_watts_strogatz(n: int, k: int, beta: float = 0.1, device='gpu'):
        """GPU最適化Watts-Strogatzネットワーク"""
        if device == 'gpu' and GPU_AVAILABLE:
            xp = cp
        else:
            xp = np
        
        # 初期リングラティス
        adjacency = xp.zeros((n, n), dtype=xp.float32)
        
        # 各ノードをk/2個の最近隣と接続
        for i in range(n):
            for j in range(1, k//2 + 1):
                right = (i + j) % n
                left = (i - j) % n
                adjacency[i, right] = 1.0
                adjacency[i, left] = 1.0
        
        # 再配線（ベクトル化）
        rewire_mask = xp.random.rand(n, n) < beta
        rewire_mask = rewire_mask & (adjacency > 0)  # 既存エッジのみ
        
        # 再配線の実行
        rewire_indices = xp.where(rewire_mask)
        if len(rewire_indices[0]) > 0:
            # 新しい接続先をランダムに選択
            new_targets = xp.random.randint(0, n, size=len(rewire_indices[0]))
            
            # 元の接続を削除
            adjacency[rewire_indices] = 0
            # 新しい接続を追加
            adjacency[rewire_indices[0], new_targets] = 1.0
        
        # 対称化
        adjacency = (adjacency + adjacency.T) > 0
        adjacency = adjacency.astype(xp.float32)
        
        return adjacency
    
    @staticmethod
    def gpu_barabasi_albert(n: int, m: int, device='gpu'):
        """GPU最適化Barabási-Albertネットワーク"""
        if device == 'gpu' and GPU_AVAILABLE:
            xp = cp
        else:
            xp = np
        
        adjacency = xp.zeros((n, n), dtype=xp.float32)
        degrees = xp.zeros(n, dtype=xp.float32)
        
        # 初期完全グラフ
        for i in range(m):
            for j in range(i+1, m):
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0
                degrees[i] += 1
                degrees[j] += 1
        
        # 優先的選択によるノード追加
        for i in range(m, n):
            # 確率に基づく選択（GPU効率化のため簡略化）
            if degrees.sum() > 0:
                probs = degrees / degrees.sum()
                # 上位m個を選択（簡略化）
                targets = xp.argsort(probs)[-m:]
                
                for target in targets:
                    if target < i:  # 自己ループ回避
                        adjacency[i, target] = 1.0
                        adjacency[target, i] = 1.0
                        degrees[i] += 1
                        degrees[target] += 1
        
        return adjacency

class GPUVotingSystem:
    """GPU最適化投票システム"""
    
    def __init__(self, agents: GPUAgent, adjacency_matrix: 'xp.ndarray'):
        self.agents = agents
        self.adjacency = adjacency_matrix
        self.n_agents = agents.n_agents
        self.xp = agents.xp
        
        # 信頼行列の初期化（隣接ノードに均等分散）
        degrees = self.adjacency.sum(axis=1)
        degrees[degrees == 0] = 1  # ゼロ除算回避
        self.trust_matrix = self.adjacency / degrees[:, None]
    
    def gpu_direct_democracy(self, signals: 'xp.ndarray') -> Tuple['xp.ndarray', 'xp.ndarray']:
        """GPU加速直接民主制"""
        # signals: (n_agents, n_issues)
        votes = signals.sum(axis=0)  # 各イシューの総投票数
        outcomes = (votes >= self.n_agents / 2).astype(self.xp.int32)
        
        # 全員が同じ重み
        weights = self.xp.ones((self.n_agents, len(outcomes)), dtype=self.xp.float32)
        
        return outcomes, weights
    
    def gpu_liquid_democracy(self, signals: 'xp.ndarray', theta: float = 0.5, 
                           lam: float = 1.0) -> Tuple['xp.ndarray', 'xp.ndarray']:
        """GPU加速液体民主制"""
        n_issues = signals.shape[1]
        
        # 委任判定（ベクトル化）
        self_vote_mask = self.agents.abilities >= theta
        
        # 委任先決定（信頼度最大のノードを選択）
        trust_scores = self.trust_matrix * self.agents.abilities[None, :].T
        delegation_targets = self.xp.argmax(trust_scores, axis=1)
        
        # 自己投票の場合は自分を指定
        delegation_targets = self.xp.where(self_vote_mask, 
                                         self.xp.arange(self.n_agents), 
                                         delegation_targets)
        
        # 委任チェーンの解決（反復法）
        current_delegates = delegation_targets.copy()
        depths = self.xp.zeros(self.n_agents, dtype=self.xp.int32)
        
        for iteration in range(10):  # 最大10回の反復
            old_delegates = current_delegates.copy()
            current_delegates = delegation_targets[current_delegates]
            depths += (current_delegates != old_delegates).astype(self.xp.int32)
            
            # 収束チェック
            if self.xp.all(current_delegates == old_delegates):
                break
        
        # 粘性重みの計算
        viscosity_weights = lam ** depths
        
        # 各最終委任先の投票重み集計
        outcomes = []
        all_weights = []
        
        for issue in range(n_issues):
            # この課題での投票
            issue_signals = signals[:, issue]
            
            # 最終委任先ごとに重みを集計
            unique_delegates = self.xp.unique(current_delegates)
            weighted_votes = self.xp.zeros(len(unique_delegates), dtype=self.xp.float32)
            delegate_votes = self.xp.zeros(len(unique_delegates), dtype=self.xp.int32)
            
            for i, delegate in enumerate(unique_delegates):
                # この委任先に委任している全エージェントのマスク
                delegate_mask = (current_delegates == delegate)
                # 重み付き投票数
                weighted_votes[i] = (viscosity_weights * delegate_mask).sum()
                # 委任先の投票
                delegate_votes[i] = issue_signals[delegate]
            
            # 重み付き多数決
            total_weighted_vote = (weighted_votes * delegate_votes).sum()
            total_weight = weighted_votes.sum()
            outcome = 1 if total_weighted_vote >= total_weight / 2 else 0
            outcomes.append(outcome)
            
            # エージェントごとの実効重み
            agent_weights = self.xp.zeros(self.n_agents, dtype=self.xp.float32)
            for i, delegate in enumerate(unique_delegates):
                delegate_mask = (current_delegates == delegate)
                agent_weights[delegate_mask] = weighted_votes[i] / delegate_mask.sum()
            
            all_weights.append(agent_weights)
        
        outcomes = self.xp.array(outcomes)
        weights = self.xp.stack(all_weights, axis=1)  # (n_agents, n_issues)
        
        return outcomes, weights

class GPUMetrics:
    """GPU最適化評価指標"""
    
    @staticmethod
    def gpu_accuracy(outcomes: 'xp.ndarray', true_answers: 'xp.ndarray', xp=cp) -> float:
        """GPU加速正答率計算"""
        correct = (outcomes == true_answers).astype(xp.float32)
        return float(correct.mean())
    
    @staticmethod
    def gpu_gini_coefficient(weights: 'xp.ndarray', xp=cp) -> 'xp.ndarray':
        """GPU加速ジニ係数計算（各イシューごと）"""
        n_agents, n_issues = weights.shape
        gini_coeffs = xp.zeros(n_issues, dtype=xp.float32)
        
        for issue in range(n_issues):
            w = weights[:, issue]
            w_sorted = xp.sort(w)
            n = len(w_sorted)
            cumsum = xp.cumsum(w_sorted)
            
            if w_sorted.sum() > 0:
                index = xp.arange(1, n + 1)
                gini = (2 * (index * w_sorted).sum() - (n + 1) * w_sorted.sum()) / (n * w_sorted.sum())
                gini_coeffs[issue] = gini
        
        return gini_coeffs

class GPUSimulator:
    """GPU最適化シミュレーター"""
    
    def __init__(self, device='gpu'):
        self.device = device
        if device == 'gpu' and GPU_AVAILABLE:
            self.xp = cp
            print(f"Using GPU: {cp.cuda.get_device_name()}")
        else:
            self.xp = np
            print("Using CPU")
    
    def run_batch_experiment(self, n_agents: int = 10000, n_issues: int = 1000, 
                           network_type: str = 'watts_strogatz', **kwargs) -> Dict:
        """大規模バッチ実験"""
        print(f"Running experiment: {n_agents} agents, {n_issues} issues")
        
        start_time = time.time()
        
        # エージェント生成
        agents = GPUAgent(n_agents, device=self.device)
        setup_time = time.time()
        print(f"Agent setup: {setup_time - start_time:.2f}s")
        
        # ネットワーク生成
        if network_type == 'watts_strogatz':
            adjacency = GPUNetworkGenerator.gpu_watts_strogatz(
                n_agents, k=kwargs.get('k', 10), beta=kwargs.get('beta', 0.1), 
                device=self.device
            )
        elif network_type == 'barabasi_albert':
            adjacency = GPUNetworkGenerator.gpu_barabasi_albert(
                n_agents, m=kwargs.get('m', 5), device=self.device
            )
        else:
            raise ValueError("Unsupported network type")
        
        network_time = time.time()
        print(f"Network generation: {network_time - setup_time:.2f}s")
        
        # 投票システム初期化
        voting_system = GPUVotingSystem(agents, adjacency)
        
        # 真の答えを生成
        true_answers = self.xp.random.choice([0, 1], size=n_issues).astype(self.xp.int32)
        
        # シグナル生成（全エージェント×全イシューを一括処理）
        signals = agents.batch_signal_generation(true_answers)
        signal_time = time.time()
        print(f"Signal generation: {signal_time - network_time:.2f}s")
        
        # 投票実行
        results = {}
        
        # 直接民主制
        direct_outcomes, direct_weights = voting_system.gpu_direct_democracy(signals)
        direct_time = time.time()
        print(f"Direct democracy: {direct_time - signal_time:.2f}s")
        
        # 液体民主制
        liquid_outcomes, liquid_weights = voting_system.gpu_liquid_democracy(
            signals, theta=0.5, lam=1.0
        )
        liquid_time = time.time()
        print(f"Liquid democracy: {liquid_time - direct_time:.2f}s")
        
        # 粘性液体民主制
        viscous_outcomes, viscous_weights = voting_system.gpu_liquid_democracy(
            signals, theta=0.5, lam=0.8
        )
        viscous_time = time.time()
        print(f"Viscous liquid democracy: {viscous_time - liquid_time:.2f}s")
        
        # 評価指標計算
        results = {
            'direct': {
                'accuracy': GPUMetrics.gpu_accuracy(direct_outcomes, true_answers, self.xp),
                'avg_gini': float(GPUMetrics.gpu_gini_coefficient(direct_weights, self.xp).mean()),
            },
            'liquid': {
                'accuracy': GPUMetrics.gpu_accuracy(liquid_outcomes, true_answers, self.xp),
                'avg_gini': float(GPUMetrics.gpu_gini_coefficient(liquid_weights, self.xp).mean()),
            },
            'viscous_liquid': {
                'accuracy': GPUMetrics.gpu_accuracy(viscous_outcomes, true_answers, self.xp),
                'avg_gini': float(GPUMetrics.gpu_gini_coefficient(viscous_weights, self.xp).mean()),
            }
        }
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\nTotal execution time: {total_time:.2f}s")
        print(f"Issues per second: {n_issues / total_time:.1f}")
        print(f"Agent-issues per second: {n_agents * n_issues / total_time:.0f}")
        
        return {
            'n_agents': n_agents,
            'n_issues': n_issues,
            'execution_time': total_time,
            'results': results
        }

def benchmark_gpu_vs_cpu():
    """GPU vs CPU性能比較"""
    print("=== GPU vs CPU Benchmark ===")
    
    test_configs = [
        {'n_agents': 1000, 'n_issues': 100},
        {'n_agents': 5000, 'n_issues': 200},
        {'n_agents': 10000, 'n_issues': 500},
    ]
    
    for config in test_configs:
        print(f"\nTesting: {config['n_agents']} agents, {config['n_issues']} issues")
        
        # CPU実行
        cpu_sim = GPUSimulator(device='cpu')
        cpu_result = cpu_sim.run_batch_experiment(**config, network_type='watts_strogatz', k=10)
        cpu_time = cpu_result['execution_time']
        
        # GPU実行（利用可能な場合）
        if GPU_AVAILABLE:
            gpu_sim = GPUSimulator(device='gpu')
            gpu_result = gpu_sim.run_batch_experiment(**config, network_type='watts_strogatz', k=10)
            gpu_time = gpu_result['execution_time']
            
            speedup = cpu_time / gpu_time
            print(f"CPU time: {cpu_time:.2f}s, GPU time: {gpu_time:.2f}s, Speedup: {speedup:.1f}x")
        else:
            print(f"CPU time: {cpu_time:.2f}s (GPU not available)")

def main():
    parser = argparse.ArgumentParser(description='GPU-Accelerated Liquid Democracy Simulation')
    parser.add_argument('--n_agents', type=int, default=10000, help='Number of agents')
    parser.add_argument('--n_issues', type=int, default=1000, help='Number of issues')
    parser.add_argument('--device', type=str, default='gpu', choices=['gpu', 'cpu'])
    parser.add_argument('--benchmark', action='store_true', help='Run GPU vs CPU benchmark')
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_gpu_vs_cpu()
    else:
        simulator = GPUSimulator(device=args.device)
        result = simulator.run_batch_experiment(
            n_agents=args.n_agents,
            n_issues=args.n_issues,
            network_type='watts_strogatz',
            k=10, beta=0.1
        )
        
        print("\n=== Results ===")
        for system, metrics in result['results'].items():
            print(f"{system.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Avg Gini: {metrics['avg_gini']:.3f}")

if __name__ == "__main__":
    main()