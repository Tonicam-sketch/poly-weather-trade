# poly-weather-trade

Polymarket 天气市场套利的**数据采集 + 回测系统**。这是把开源天气套利策略改造成"长期盈利、控回撤"的自动交易系统的第一步：在投入任何真金白银之前，先证明 edge 是真的。

研究层用 Python（校准、概率建模、回测、指标分析），未来的下单/签名执行层保持 TypeScript，两层通过 SQLite 解耦。

## 为什么先做这个

原开源策略直接拿集合预报的原始频率当概率、用 5% 纸面 edge 下注、让 LLM 算钱。三个致命问题：(1) 原始集合**欠离散**，概率被高估，edge 多半是假的；(2) edge 没扣点差/费用，纸面 5% 会被吃光；(3) 钱的计算交给 LLM，非确定、不可审计。本系统针对性地解决这些：

- **校准优先**：方差膨胀修正欠离散（`fit_variance_inflation`）+ isotonic 概率校准 + Brier/可靠性曲线。
- **净 edge**：按可成交的 ask 价、扣点差和费用计算（`model/edge.py`）；超大 edge 当作数据 bug 告警而非信号。
- **确定性内核**：所有概率、edge、Kelly、PnL、风控都是带单元测试的纯函数，零 LLM 参与。
- **可复现存储**：单一 SQLite 库，每条预报/价格/结算都带时间戳，取代原策略并发读写一堆 JSON 的做法。

## 架构

```
src/pwt/
  datasources/   API 客户端：Open-Meteo（集合预报/历史/ERA5）、Polymarket（Gamma/CLOB）
  collect/       编排层：把数据拉进 SQLite（需要网络）
  storage/       SQLite 持久化（schema + 读写）
  model/         确定性量化内核
    units.py       摄氏/华氏换算（预报存摄氏，市场区间用原生单位）
    ensemble.py    集合成员 → 区间概率（直方图 / 高斯-EMOS）
    calibration.py 方差膨胀、isotonic、Brier、可靠性曲线
    edge.py        成本模型 + 净 edge + 过滤（含告警阈值）
    sizing.py      收缩 + 分数 Kelly + 仓位/流动性上限
  backtest/
    engine.py      run_backtest（纯函数）+ build_evaluations_from_db（数据对齐）
    metrics.py     夏普、最大回撤、Calmar、胜率、盈亏比
  cli.py         命令行入口
config/cities.yaml  城市 → 经纬度/时区/结算站点/单位
tests/              38 个单元测试，全部离线可跑
```

## 安装

```bash
pip install -e .          # 或：pip install numpy pandas scipy scikit-learn requests pyyaml pyarrow
pip install -e ".[dev]"   # 含 pytest
```

## 使用

```bash
pwt init-db                                               # 初始化 SQLite
pwt collect-weather --issue-date 2025-06-01 --horizon 5   # 拉某天发布的集合预报
pwt collect-actuals  --start 2025-06-01 --end 2025-06-30  # 拉 ERA5 实况（结算真值代理）
pwt collect-markets                                       # 拉已结算的天气市场 + 价格历史
pwt status                                                # 查看各表行数
pwt backtest --lead-days 1 --inflation 1.0 --capital 1000 # 跑回测
```

数据目录和库路径可用环境变量覆盖：`PWT_DATA_DIR`、`PWT_DB_PATH`。

### ⚠ 网络白名单

采集命令需要访问以下 host，请确认它们在运行环境的 egress 白名单里（否则会报 `Host not in allowlist`）：

```
ensemble-api.open-meteo.com
historical-forecast-api.open-meteo.com
archive-api.open-meteo.com
gamma-api.polymarket.com
clob.polymarket.com
```

确定性内核和全部测试无需网络。

## 测试

```bash
pytest -q     # 38 passed
```

## 已知的简化（下一步）

- **结算真值用 ERA5 代理**，生产应换成 Polymarket 实际结算的站点观测（Meteostat/GHCN）。站点对齐比预报本身更重要——合约结算在某个具体站点。
- **历史盘口深度只有中间价**，成本模型用假设点差；接入 CLOB order book 后可做真实滑点/容量约束。
- **市场的城市/日期靠问题文本启发式匹配**，覆盖前应人工抽检。
- 回测的组合风控（每日新仓上限、日亏熔断、回撤 kill-switch）是对实盘并发持仓的近似。
