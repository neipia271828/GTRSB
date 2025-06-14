import unittest
import numpy as np
from requid_dem_model import GPUAgent, GPUNetworkGenerator, GPUVotingSystem, GPUMetrics

class TestRequidDemModel(unittest.TestCase):
    def setUp(self):
        """テストの前準備"""
        self.n_agents = 100
        self.agent = GPUAgent(self.n_agents, device='cpu')
        self.network = GPUNetworkGenerator.gpu_watts_strogatz(self.n_agents, k=4, beta=0.1, device='cpu')
        self.voting_system = GPUVotingSystem(self.agent, self.network)

    def test_gpu_agent_initialization(self):
        """GPUAgentの初期化テスト"""
        self.assertEqual(self.agent.n_agents, self.n_agents)
        self.assertEqual(len(self.agent.abilities), self.n_agents)
        self.assertEqual(len(self.agent.ideologies), self.n_agents)
        self.assertEqual(self.agent.trust_matrix.shape, (self.n_agents, self.n_agents))

    def test_batch_signal_generation(self):
        """バッチシグナル生成のテスト"""
        true_answers = np.array([1, 0, 1])
        signals = self.agent.batch_signal_generation(true_answers)
        self.assertEqual(signals.shape, (self.n_agents, len(true_answers)))
        self.assertTrue(np.all(np.isin(signals, [0, 1])))

    def test_network_generation(self):
        """ネットワーク生成のテスト"""
        # Watts-Strogatz
        ws_network = GPUNetworkGenerator.gpu_watts_strogatz(self.n_agents, k=4, beta=0.1, device='cpu')
        self.assertEqual(ws_network.shape, (self.n_agents, self.n_agents))
        self.assertTrue(np.all(np.isin(ws_network, [0, 1])))
        
        # Barabási-Albert
        ba_network = GPUNetworkGenerator.gpu_barabasi_albert(self.n_agents, m=2, device='cpu')
        self.assertEqual(ba_network.shape, (self.n_agents, self.n_agents))
        self.assertTrue(np.all(np.isin(ba_network, [0, 1])))

    def test_direct_democracy(self):
        """直接民主制のテスト"""
        signals = np.random.randint(0, 2, (self.n_agents, 3))
        outcomes, weights = self.voting_system.gpu_direct_democracy(signals)
        self.assertEqual(len(outcomes), 3)
        self.assertEqual(weights.shape, (self.n_agents, 3))
        self.assertTrue(np.all(np.isin(outcomes, [0, 1])))

    def test_liquid_democracy(self):
        """液体民主制のテスト"""
        signals = np.random.randint(0, 2, (self.n_agents, 3))
        outcomes, weights = self.voting_system.gpu_liquid_democracy(signals, theta=0.5, lam=1.0)
        self.assertEqual(len(outcomes), 3)
        self.assertEqual(weights.shape, (self.n_agents, 3))
        self.assertTrue(np.all(np.isin(outcomes, [0, 1])))

    def test_edge_cases(self):
        """エッジケースのテスト"""
        # 1人のエージェント
        single_agent = GPUAgent(1, device='cpu')
        single_network = GPUNetworkGenerator.gpu_watts_strogatz(1, k=2, beta=0.1, device='cpu')
        single_voting = GPUVotingSystem(single_agent, single_network)
        
        signals = np.array([[1]])
        outcomes, weights = single_voting.gpu_liquid_democracy(signals)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(weights.shape, (1, 1))

        # 全員同じ投票
        uniform_signals = np.ones((self.n_agents, 3))
        outcomes, weights = self.voting_system.gpu_liquid_democracy(uniform_signals)
        self.assertTrue(np.all(outcomes == 1))

        # 全員異なる投票
        alternating_signals = np.tile([0, 1], (self.n_agents, 2))
        outcomes, weights = self.voting_system.gpu_liquid_democracy(alternating_signals)
        self.assertEqual(len(outcomes), 4)

    def test_metrics(self):
        """評価指標のテスト"""
        # テスト用の重みを生成（n_agents x n_issues）
        weights = np.random.rand(self.n_agents, 3)
        gini_coeffs = GPUMetrics.gpu_gini_coefficient(weights)
        
        # 各イシューのジニ係数が0から1の間であることを確認
        self.assertEqual(len(gini_coeffs), 3)  # 3つのイシュー
        self.assertTrue(np.all((gini_coeffs >= 0) & (gini_coeffs <= 1)))
        
        # 完全平等な場合のジニ係数が0であることを確認
        equal_weights = np.ones((self.n_agents, 1))
        equal_gini = GPUMetrics.gpu_gini_coefficient(equal_weights)
        self.assertAlmostEqual(equal_gini[0], 0.0, places=6)
        
        # 完全不平等な場合のジニ係数が理論値に一致することを確認
        unequal_weights = np.zeros((self.n_agents, 1))
        unequal_weights[0] = 1.0  # 1人だけが全ての重みを持つ
        unequal_gini = GPUMetrics.gpu_gini_coefficient(unequal_weights)
        theoretical_max = (self.n_agents - 1) / self.n_agents
        self.assertAlmostEqual(unequal_gini[0], theoretical_max, places=6)

if __name__ == '__main__':
    unittest.main() 