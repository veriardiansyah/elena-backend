import numpy as np

class QLearningNetworkAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((3, 4))
        
        # Hyperparameters Reinforcement Learning
        self.alpha = alpha      # Learning Rate
        self.gamma = gamma      # Discount Factor
        self.epsilon = epsilon  # Exploration Rate
        
        # Representasi Aksi/Rekomendasi yang dapat dipilih oleh Agen RL
        self.actions = [
            "Koneksi Stabil. Tidak ada tindakan diperlukan.",
            "Periksa koneksi ISP Anda atau hubungi customer service.",
            "Kurangi penggunaan bandwidth (Batasi streaming/unduhan besar).",
            "Lakukan traffic balancing atau beralih ke jalur cadangan (Backup Link)."
        ]
        
        # Bootstrapping Knowledge 
        self.q_table[0][0] = 5.0  # Baik -> No Action
        self.q_table[1][2] = 4.0  # Normal -> Kurangi Bandwidth
        self.q_table[2][1] = 4.5  # Buruk -> Cek ISP
        self.q_table[2][3] = 4.8  # Buruk -> Traffic Balancing

    def get_state(self, latency, packet_loss):
        """Fungsi Kuantisasi/Fuzzy sederhana memetakan metrik ke nomor State"""
        if latency > 150 or packet_loss > 5:
            return 2  # State Buruk
        elif 50 <= latency <= 150 or 1 <= packet_loss <= 5:
            return 1  # State Normal
        else:
            return 0  # State Baik

    def calculate_reward(self, state, action_idx):
        """Reward Function: Memberikan feedback nilai ke Agen"""
        if state == 0 and action_idx == 0: return 10.0
        if state == 1 and action_idx == 2: return 10.0
        if state == 2 and (action_idx == 1 or action_idx == 3): return 10.0
        return -5.0  # Penalti jika keputusan agen tidak tepat/ngawur

    def get_recommendation_and_learn(self, latency, packet_loss):
        """Fungsi Inti: Mengambil aksi berdasarkan Q-Table dan mengupdate nilai Q"""
        state = self.get_state(latency, packet_loss)
        
        # Strategi Epsilon-Greedy
        if np.random.uniform(0, 1) < self.epsilon:
            action_idx = np.random.choice(len(self.actions)) # Eksplorasi
        else:
            action_idx = np.argmax(self.q_table[state]) # Eksploitasi
            
        reward = self.calculate_reward(state, action_idx)
        next_state = state  # Diasumsikan steady state pada siklus pendek
        
        # Persamaan Bellman untuk update nilai matriks Q-Table
        old_value = self.q_table[state][action_idx]
        next_max = np.max(self.q_table[next_state])
        self.q_table[state][action_idx] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        
        state_labels = {0: "BAIK", 1: "NORMAL", 2: "BURUK"}
        return state_labels[state], self.actions[action_idx]