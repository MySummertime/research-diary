# 10 classic risk evaluation models

![10 risk models](../Attachments/img/risk%20models.png "10 risk evaluation models")

## 1. 传统风险模型 Traditional Risk (TR)
Source:
- Alp 提出的 TR 模型旨在最小化危险货物卡车沿着路径运输过程中产生不良后果的期望值；  
    [Alp E. Risk-based transportation planning practice: Overall methodology and a case example［J］．Information Systems and Operational Ｒesearch，1995，33(1): 4－19．](https://www.researchgate.net/publication/259716076_Risk-Based_Transportation_Planning_Practice_Overall_Methodology_And_A_Case_Example)
- Jin 等对其做了近似，最终得到表中的 TR 模型.  
  [Jin H，Batta R．Objectives derived from viewing hazmat shipments as a sequence of independent Bernoulli trials［J］．Transportation Science，1997，31(3): 252－261.](https://doi.org/10.1287/trsc.31.3.252)

## 2. 事故概率模型 Incident Probability (IP)
Source:
- Saccomanno 等提出了 IP 模型。  
  [Saccomanno F，Chan A．Economic evaluation of routing strategies for hazardous road shipments［J］. Transportation Research Record，1986(1 020):12－18．](https://onlinepubs.trb.org/Onlinepubs/trr/1985/1020/1020-003.pdf)

Feature:
- IP 模型是 TR 模型的极端情况，只关注事故概率。

## 3. 人口暴露模型 Population Exposure (PE)
Source:
- ReVelle 等提出了 PE 模型。  
  [ReVelle C，Cohon J，Shobrys D．Simultaneous sitting and routing in the disposal of hazardous wastes［J］．Transportation Science，1991，25(2): 138－145．](https://pubsonline.informs.org/doi/10.1287/trsc.25.2.138)

Feature:
- PE 模型是 TR 模型的极端情况，只关注运输影响区域内的总后果。

## 4. 感知风险模型 Perceived Risk (PR)
Source:
- Abkowitz 等建立了 PR 模型。  
  [Abkowitz M，Lepofsky M，Cheng P．Selecting criteria for designating hazardous materials highway routes［J］．Transportation Research Record，1992，1333: 30－35．](https://onlinepubs.trb.org/Onlinepubs/trr/1992/1333/1333-005.pdf)

Feature:
- PR 模型通过添加**权重参数**来反映决策者对风险的偏好，在一定程度上可避免决策者的风险中性问题。

## 5. 均值-方差模型 Mean-Variance (MV)
Source:
- Erkut 等建立了 MV 模型。  
  [Erkut E，Ingolfsson A．Catastrophe avoidance models for hazardous materials route planning［J］．Transportation Science，2000，34(2): 165－179．](https://doi.org/10.1287/trsc.34.2.165.12303)

Feature:
- MV 模型引入了效用理论，以期避免大灾难。

## 6. 负效用模型 Disutility (DU)
Source:
- Erkut 等建立了 DU 模型。  
  [Erkut E，Ingolfsson A．Catastrophe avoidance models for hazardous materials route planning［J］．Transportation Science，2000，34(2): 165－179．](https://doi.org/10.1287/trsc.34.2.165.12303)

Feature:
- DU 模型旨在找到拥有最小的均值和方差的路径，以期避免大灾难。

## 7. 最小最大后果模型 Minimized Maximum risk (MM)
Source:
- Erkut 等建立了 MM 模型。  
  [Erkut E，Ingolfsson A．Catastrophe avoidance models for hazardous materials route planning［J］．Transportation Science，2000，34(2): 165－179．](https://doi.org/10.1287/trsc.34.2.165.12303)

Feature:
- MM 模型旨在通过最小化整个运输过程中的最大风险来避免大灾难。

## 8. 条件概率风险模型 Conditional probability Risk (CR)
Source:
- Sivakumar 等提出了 CR 模型。  
  [Sivakumar R，Batta R，Karwan M．A multiple route conditional risk model for transporting hazardous materials［J］．Information Systems and Operational Ｒesearch，1995，33(1):20－33．](https://doi.org/10.1080/03155986.1995.11732264)

Feature:
- CR 模型同时具有 TR 和 IP 这两个属性，目标函数不可加。

## 9. 风险价值模型 Value-at-Risk (VaR)
Source:
- KANG 等将 VaR 模型引入危险货物运输领域。  
  [KANG Ying-ying．Value-at-risk models for hazardous materials transportation［D］．New York: State University of New York at Buffalo，2011．](https://www.acsu.buffalo.edu/~batta/hazmatvar.pdf)

Feature:
- 进行路径决策时，VaR 模型可生成多条路径，以满足不同决策者的风险偏好。

## 10. 条件风险价值模型 Conditional Value-at-Risk (CVaR)
Source:
- Kwon 等将 CVaR 模型引入危险货物运输领域。  
  [Kwon C．Conditional value-at-risk model for hazardous materials transportation［C］．Proceedings of the 2011 Winter Simulation Conference，2011: 1703－1709．](https://www.informs-sim.org/wsc11papers/152.pdf)

Feature:
- 进行路径决策时，CVaR 模型可生成多条路径，以满足不同决策者的风险偏好。