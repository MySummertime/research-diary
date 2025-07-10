---
layout: default
title: Risk Evaluation Models
date: 2025-4-21
---

<small>{{ page.date | date: "%-d %B %Y" }}</small>

# 10 classic risk evaluation models

![10 risk models](../assets/images/risk%20models.png "10 risk evaluation models")

## 1. 传统风险模型 Traditional Risk (TR)

Source:

- Alp 提出的 TR 模型旨在最小化危险货物卡车沿着路径运输过程中产生不良后果的期望值；  
  [[1] Alp E. Risk-based transportation planning practice: Overall methodology and a case example[J]. Information Systems and Operational Research, 1995, 33(1): 4-19.](https://www.researchgate.net/publication/259716076_Risk-Based_Transportation_Planning_Practice_Overall_Methodology_And_A_Case_Example)
- 由于该模型计算复杂，Jin 等对其进行了近似计算，最终得到表中的 TR 模型。  
  [[2] Jin H, Batta R. Objectives derived from viewing hazmat shipments as a sequence of independent Bernoulli trials[J]. Transportation Science, 1997, 31(3): 252-261.](https://doi.org/10.1287/trsc.31.3.252)

Feature:

- TR 模型具有 risk neutral 偏好，直接估算出事故后果的期望值。
- 其精确模型可能单调、非凸，违反了路径单调性原则和属性单调性原则；
- 其近似模型更易于优化。

## 2. 事故概率模型 Incident Probability (IP)

Source:

- Saccomanno 等提出了 IP 模型。  
  [[3] Saccomanno F, Chan A. Economic evaluation of routing strategies for hazardous road shipments[J]. Transportation Research Record, 1986(1020): 12-18.](https://onlinepubs.trb.org/Onlinepubs/trr/1985/1020/1020-003.pdf)

Feature:

- IP 模型是 TR 模型的极端情况，只关注事故概率。

## 3. 人口暴露模型 Population Exposure (PE)

Source:

- ReVelle 等提出了 PE 模型。  
  [[4] ReVelle C, Cohon J, Shobrys D. Simultaneous sitting and routing in the disposal of hazardous wastes[J]. Transportation Science, 1991, 25(2): 138-145.](https://pubsonline.informs.org/doi/10.1287/trsc.25.2.138)

Feature:

- PE 模型是 TR 模型的极端情况，只关注运输影响区域内的总后果，后果可以用人口密度和受影响面积的乘积来表示。

TR, IP, PE 是风险中性模型，因此经常被危险品运输公司所采用。  
然而，在实际场景中，对危险品运输进行规划和决策的风险决策者还包括：政府部门、居民。  
因此，在多决策者情景中，以上 3 种模型并不适用。  
PR, MV, DU, MM, CR 都是风险厌恶/规避模型。

## 4. 感知风险模型 Perceived Risk (PR)

Source:

- Abkowitz 等建立了 PR 模型。  
  [[5] Abkowitz M, Lepofsky M, Cheng P. Selecting criteria for designating hazardous materials highway routes[J]. Transportation Research Record, 1992, 1333: 30-35.](https://onlinepubs.trb.org/Onlinepubs/trr/1992/1333/1333-005.pdf)

Feature:

- PR 模型通过添加 **权重参数q** 来反映决策者对风险的偏好，在一定程度上可避免决策者的风险中性问题。

以下 3 个模型：MV, DU, MM，均适用于需要避免大灾难的情况。

## 5. 均值-方差模型 Mean-Variance (MV)

Source:

- Erkut 等建立了 MV 模型。  
  [[6] Erkut E, Ingolfsson A. Catastrophe avoidance models for hazardous materials route planning[J]. Transportation Science, 2000, 34(2): 165-179.](https://doi.org/10.1287/trsc.34.2.165.12303)

Feature:

- MV 模型引入了效用理论 the theory of utility，旨在寻找均值和方差最小的路径。

## 6. 负效用模型 Disutility (DU)

Source:

- Erkut 等建立了 DU 模型。  
  [[6] Erkut E, Ingolfsson A. Catastrophe avoidance models for hazardous materials route planning[J]. Transportation Science, 2000, 34(2): 165-179.](https://doi.org/10.1287/trsc.34.2.165.12303)

Feature:

- DU 模型同样基于效用理论，也旨在找到均值和方差最小的路径。

## 7. 最小最大后果模型 Minimized Maximum risk (MM)

Source:

- Erkut 等建立了 MM 模型。  
  [[6] Erkut E, Ingolfsson A. Catastrophe avoidance models for hazardous materials route planning[J]. Transportation Science, 2000, 34(2): 165-179.](https://doi.org/10.1287/trsc.34.2.165.12303)

Feature:

- MM 模型旨在通过最小化整个运输过程中的最大风险来避免大灾难。

以上 7 个模型都是单一属性的 single attribute，且**目标函数均有可加性**。

## 8. 条件概率风险模型 Conditional probability Risk (CR)

Source:

- Sivakumar 等提出了 CR 模型。  
  [[7] Sivakumar R, Batta R, Karwan M. A multiple route conditional risk model for transporting hazardous materials[J]. Information Systems and Operational Research, 1995, 33(1): 20-33.](https://doi.org/10.1080/03155986.1995.11732264)

Feature:

- CR 模型结合了 TR 和 IP 模型的属性，且**目标函数不可加**。
- non-additive 的含义：  
  给定路径上的总风险不是通过简单地将各路段的风险相加来计算的。  
  根据 CR 模型的公式来看，给定路径的总风险是一个比率，具体来说是所有路段上的风险之和的加权平均值，其中权重是所有路段上的事故概率。  
  这种不可加性意味着，如果增加路径中某个环节的事故概率，反而会降低给定路径上的总风险。

使用以上 8 个模型进行路径决策时，只能得到一条路径，且没有明确考虑决策者的风险偏好。  
以下 2 个模型：VaR, CVaR，既能满足不同决策者的风险偏好要求，又能解决其他模型的缺点，如缺乏可扩展性和表达性。

## 9. 风险价值模型 Value-at-Risk (VaR)

Source:

- KANG 等将 VaR 模型引入危险货物运输领域。  
  [[8] ANG Ying-ying. Value-at-risk models for hazardous materials transportation[D]. New York: State University of New York at Buffalo, 2011.](https://www.acsu.buffalo.edu/~batta/hazmatvar.pdf)

Feature:

- VaR 模型可以通过改变 conficence level 来满足 risk unconcered, risk neutral, risk aversion 等偏好。
- 可作为非离散、非凸的函数衡量风险，一般计算较为复杂。
- 进行路径决策时，VaR 模型可生成多条路径。
- 然而，它只关注不超过 VaR 值的风险，无法控制超过 VaR 值的风险，即尾部风险。
- 适用于决策者有特定的风险承受水平，且尾部模型不可靠时。

## 10. 条件风险价值模型 Conditional Value-at-Risk (CVaR)

Source:

- Kwon 等将 CVaR 模型引入危险货物运输领域。  
  [[9] Kwon C. Conditional value-at-risk model for hazardous materials transportation[C]. Proceedings of the 2011 Winter Simulation Conference, 2011: 1703-1709.](https://www.informs-sim.org/wsc11papers/152.pdf)

Feature:

- CVaR 模型天然具有极端的 risk aversion 偏好。
- 一般是连续的凸函数，通常更容易计算和优化，可转化为线性规划、凸优化等。
- 进行路径决策时，CVaR 模型可生成多条路径。
- 适用于估算极端尾部损失，或需要更鲁棒和全面的风险控制时。
