import numpy as np
import matplotlib.pyplot as plt
from rich.progress import track

############################################
# 参数设置（减少迭代数、传输次数加快运行）
############################################
N = 200         # LDPC码长
M = 100         # 校验方程数量（M < N）
num_runs = 100   # 每个SNR下的码字传输数量（减少以加快运行）
SNR_dB_list = [0, 1, 2, 3, 4, 5]  # 测试的SNR列表(单位dB)
iterations = 1000      # MCMC总迭代次数（减少）
burn_in = 100          # MCMC烧入期（减少）

np.random.seed(42)

############################################
# 构造随机稀疏LDPC校验矩阵H (M x N)
############################################
density = 0.1
H = np.zeros((M, N), dtype=int)
for i in range(M):
    ones_positions = np.random.choice(N, int(N*density), replace=False)
    H[i, ones_positions] = 1

############################################
# 固定码字为全0码字，满足Hx=0，无需重复尝试
############################################
def generate_codeword():
    return np.zeros(N, dtype=int)

############################################
# MCMC相关函数定义
############################################
def pi_func(x, y, sigma2):
    # p(x|y) ∝ p(y|x)*p(x)
    # 先验p(x)均匀
    # p(y|x)=exp(-||y - x||^2/(2*sigma²))
    # 不满足Hx=0的x, 概率极低。
    if not np.all((H.dot(x) % 2) == 0):
        return 1e-30
    diff = y - x
    return np.exp(-np.sum(diff**2)/(2*sigma2))

def mcmc_decode(y, sigma2, iterations=1000, burn_in=200):
    # MCMC解码：Metropolis-Hastings
    x_current = (y > 0.5).astype(int)  # 根据接收值初始化
    current_prob = pi_func(x_current, y, sigma2)

    samples = []
    prob_trace = []

    for t in range(iterations):
        i = np.random.randint(N)
        x_proposal = x_current.copy()
        x_proposal[i] = 1 - x_proposal[i]

        proposal_prob = pi_func(x_proposal, y, sigma2)
        alpha = min(1, proposal_prob/current_prob)

        if np.random.rand() < alpha:
            x_current = x_proposal
            current_prob = proposal_prob

        prob_trace.append(current_prob)

        if t > burn_in:
            samples.append(x_current.copy())
    
    samples = np.array(samples)
    posterior_prob = np.mean(samples, axis=0) if len(samples) > 0 else x_current.astype(float)
    return posterior_prob, prob_trace

############################################
# 主仿真循环：对不同SNR测试BER性能
############################################
BER_list = []
convergence_info = None

for SNR_dB in track(SNR_dB_list, description="Processing SNR points..."):
    SNR = 10**(SNR_dB/10)
    sigma2 = 1/(2*SNR)  # BPSK情况下的噪声方差
    errors = 0
    total_bits = 0

    # 对每个SNR进行多次传输以求平均BER
    for run_idx in range(num_runs):
        # 生成码字（全0）
        x_true = generate_codeword()
        # BPSK调制：0->-1, 1->+1
        x_bpsk = 2*x_true - 1
        noise = np.sqrt(sigma2)*np.random.randn(N)
        y = x_bpsk + noise

        # 将y映射回[0,1]范围，方便pi_func使用
        y_for_pi = (y + 1)/2

        # MCMC解码
        posterior_prob, prob_trace = mcmc_decode(y_for_pi, sigma2, iterations=iterations, burn_in=burn_in)
        x_hat = (posterior_prob > 0.5).astype(int)
        bit_errors = np.sum(x_hat != x_true)
        errors += bit_errors
        total_bits += N

        # 保存一次收敛信息用于后续绘图（仅在一个中间SNR和第一个run中）
        if SNR_dB == SNR_dB_list[len(SNR_dB_list)//2] and run_idx == 0:
            convergence_info = (prob_trace, posterior_prob, x_true, x_hat)

    BER = errors / total_bits
    BER_list.append(BER)

############################################
# 绘制BER曲线
############################################
plt.figure(figsize=(8,6))
plt.semilogy(SNR_dB_list, BER_list, marker='o')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER vs SNR (MCMC-based decoding)')
plt.grid(True)
plt.savefig('BER_curve.png', dpi=300)
plt.show()

############################################
# 绘制MCMC收敛特性图像
############################################
if convergence_info is not None:
    prob_trace, posterior_prob, x_true, x_hat = convergence_info
    plt.figure(figsize=(8,6))
    plt.plot(prob_trace)
    plt.xlabel('Iteration')
    plt.ylabel('Posterior Probability (unnormalized)')
    plt.title('MCMC Convergence Trace')
    plt.grid(True)
    plt.savefig('MCMC_convergence.png', dpi=300)
    plt.show()

    # 打印前10位结果对比
    print("True bits   :", x_true[:10])
    print("Decoded bits:", x_hat[:10])
    print("Posterior prob of first 10 bits:", posterior_prob[:10])
    print("BER at selected scenario:", np.mean(x_hat != x_true))
