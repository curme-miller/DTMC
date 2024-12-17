## 功能概述
- 构造随机LDPC校验矩阵H。
- 随机生成满足Hx=0约束的码字x。
- 将码字通过AWGN信道传输，并接收信号y。
- 使用MCMC方法对后验分布p(x|y)进行抽样逼近，以估计各比特为1的概率并进行软判决译码。
- 多次仿真以在不同SNR下统计误码率（BER）。
- 绘制BER随SNR变化的曲线。
- 绘制MCMC收敛图像，如后验概率评估值（或接收率）随迭代次数的变化。

## 代码说明
- 此代码使用一个随机生成的LDPC校验矩阵H，以及简单的MCMC方法对码字进行后验近似。在实际LDPC系统中，应使用标准LDPC码（如IEEE 802.11n或5G标准中定义的LDPC）和更为复杂的译码策略。
- MCMC过程仅是概念演示，可能在低SNR下不能很好收敛，实际效果与参数选择、迭代次数、提议分布策略等因素有关。
- 运行此代码后，会在当前目录下保存BER曲线和MCMC收敛图像。用户可根据需要调整N、M、iterations、burn_in及SNR范围等参数。
