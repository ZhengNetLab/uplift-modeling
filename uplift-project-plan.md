# Uplift 项目计划：营销活动的精准投放

创建日期：2026-11-09｜周期：约 3 周（15 个工作日）

## 一、核心设计

**业务问题**：广告预算有限，应该投给谁？目标是找出"投了才会来、不投就不来"的用户，而不是本来就会来的用户。

**数据**：Criteo Uplift v2.1。13,979,592 行，12 个匿名连续特征（f0–f11），treatment 占约 85%，visit 基础率约 4.7%，conversion 很稀少，另有 exposure 字段（是否真正看到广告）。下载地址：http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz

**结果变量**：visit 为主，conversion 为辅（太稀少，异质性效应噪声大）。

**和普通教程的区别**：
1. 先验证实验：随机化检查、ATE 及置信区间
2. 用 exposure 做 IV：区分"被分配"（ITT）和"真正看到"的效应
3. 和 response model 对比：说明"预测谁会买"不等于"该投给谁"
4. 量化不确定性：Qini 的 bootstrap 置信区间、分组校准
5. 换算成投放策略和预算

## 二、开始前准备（Day 1）

- 环境：Python 3.11+，pandas/polars，pyarrow，lightgbm，econml，scikit-uplift，statsmodels/linearmodels，matplotlib
- 数据：下载后转成 parquet，数值列用 float32
- 仓库：数据不提交（Criteo 是研究用途条款，不能二次分发），只放下载脚本
- 阅读（挑重点）：Diemert et al. (2018) 数据论文；Künzel et al. (2019, PNAS) meta-learners；Qini 曲线定义（Radcliffe 2007 或 sklift 文档）；EconML 的 CausalForestDML 文档
- 兼容性：scikit-uplift 更新不频繁，如果和新版 scikit-learn 冲突，就只用它的评估函数，或者自己写 Qini（几十行代码，顺便证明懂原理）

## 三、分阶段计划

### 第 1 周：实验验证和平均效应
- Day 1：环境、数据、仓库
- Day 2：EDA。样本量、处理组比例、visit/conversion/exposure 基础率、特征分布
- Day 3：随机化检查。两组协变量的标准化均值差（SMD）
- Day 4：ATE。均值差加置信区间；加协变量做回归调整，看方差缩减多少
- Day 5：IV。先检查对照组 exposure 是否全为 0；如果是，属于单侧不依从，用 treatment 作 exposure 的工具变量，估计被曝光用户的效应（Bloom 估计量或 2SLS），和 ITT 对比
- 产出：notebook 01、02；ATE 汇总表

### 第 2 周：建模
- Day 6：按 treatment × visit 分层划分 train/val/test；先写好评估代码（Qini、uplift 曲线、uplift@k）
- Day 7：两个基准：随机投放；response model（只预测 P(visit)）
- Day 8：S-learner、T-learner（基模型用 LightGBM）
- Day 9：X-learner（适合 85/15 这种组间不平衡）、DR-learner（EconML）
- Day 10：CausalForestDML，在 100–200 万子样本上跑
- 产出：notebook 03；所有模型在 test set 上的 Qini 对比表

### 第 3 周：评估、业务换算、包装
- Day 11：评估加深。Qini 的 bootstrap 置信区间；按预测 uplift 分 10 组，对比每组的预测 uplift 和实际 uplift（校准图）
- Day 12：投放策略。设定单次投放成本和单次 visit 价值（假设参数），规则是"预测 uplift × 价值 > 成本才投"；画"投放比例 vs 保留的增量 visit"曲线；对成本/价值比做敏感性分析
- Day 13：异质性解释。用 SHAP 看哪些特征驱动 uplift（特征匿名，在局限里注明）
- Day 14：写 README；把 notebook 里的逻辑整理进 src/
- Day 15：用真实数字更新简历条目；写一页总结（可以同时发 LinkedIn）

## 四、仓库结构

```
criteo-uplift/
├── README.md
├── data/                 # .gitignore，不提交
├── scripts/download_data.py
├── notebooks/
│   ├── 01_eda_randomization.ipynb
│   ├── 02_ate_iv.ipynb
│   ├── 03_uplift_models.ipynb
│   └── 04_evaluation_policy.ipynb
├── src/uplift/
│   ├── metrics.py        # Qini, AUUC, uplift@k, bootstrap
│   ├── models.py
│   └── policy.py
├── reports/figures/
└── requirements.txt
```

## 五、README 结构

1. 一句话结论加一张主图（Qini 曲线或预算曲线）
2. 业务问题
3. 数据与实验验证
4. 方法和选择理由
5. 结果（换算成业务数字）
6. 局限
7. 复现步骤

## 六、风险与对策

| 风险 | 对策 |
|---|---|
| conversion 太稀少，模型几乎学不到 | visit 为主；conversion 如实报告置信区间 |
| uplift 模型只比随机投放好一点 | 在这个数据集上很常见，如实写；重点展示评估的严谨性 |
| 特征匿名，讲不出"哪类用户" | 写进局限；时间宽裕的话，加 1–2 天用 Hillstrom 数据（特征可解释）做补充 |
| 全量数据跑 causal forest 太慢 | 用子样本；meta-learner 用全量或 500 万子样本 |
| 时间超支 | Day 13 和 Hillstrom 扩展可以砍；评估和 README 不能砍 |

## 七、面试要准备的问题

- 为什么不直接投给最可能购买的人？（用自己的 response model 对比结果回答）
- treatment 占 85% 对建模有什么影响？为什么选 X-learner？
- 个体 uplift 观察不到，怎么评估模型？
- ITT 和曝光效应有什么区别？业务上该看哪个？
- 上线后怎么监控？（长期保留一个随机对照组，持续测量真实增量）

## 完成标准

- [ ] 随机化检查和 ATE（含置信区间）
- [ ] 至少 4 个 uplift 模型和 response model 基准在同一 test set 上比较
- [ ] Qini 曲线带 bootstrap 置信区间，有校准图
- [ ] 投放策略曲线和敏感性分析
- [ ] README 能让陌生人 3 分钟内看懂结论
- [ ] 简历条目替换成真实数字
