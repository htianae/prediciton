# 熔炼炉气耗 Champion

本项目仅使用历史 Excel 离线比较 LightGBM、Linear Regression 和 Huber Regression，通过五折时间滚动验证与推荐安全压力测试选择唯一 Champion。上线程序只加载 `artifacts/gas_champion.joblib`，不会重新训练。

## 训练

完整训练会让三个模型分别在五个投料重量场景、十个随机种子下公平比较 Random Search 与 GA：

```bash
python train_champion.py \
  --excel 4_month_data_2026_02_01_2026_06_25.xlsx \
  --output-dir . \
  --budget 5000 \
  --seeds 10
```

快速流程测试可以降低预算，但不能用快速结果替换正式 Champion：

```bash
python train_champion.py --excel 4_month_data_2026_02_01_2026_06_25.xlsx --budget 200 --seeds 1 --output-dir quick_check
```

## 炉次结束后预测

输入六个最终工艺值：

```bash
python predict_champion.py \
  --model artifacts/gas_champion.joblib \
  --total-weight 86000 \
  --solid-ratio 40 \
  --melting-time 9.2 \
  --waiting-time 0.8 \
  --door-open-count 13 \
  --door-open-duration 75
```

输出包含点预测、90% 预测区间、折模型不确定性和历史范围 OOD 警告。

## 参数推荐

```bash
python recommend_champion.py \
  --model artifacts/gas_champion.joblib \
  --total-weight 86000
```

也可以显式提供参考熔炼时间：

```bash
python recommend_champion.py \
  --model artifacts/gas_champion.joblib \
  --total-weight 86000 \
  --melting-time 9.0
```

推荐中的改善比例只能理解为模型估计，必须经过工厂受控试验验证，不能直接描述为已经实现的真实节气率。

## 主要产物

- `artifacts/gas_champion.joblib`：唯一上线模型、预处理、特征契约、不确定性和优化配置。
- `reports/model_comparison.csv`：三模型时间回测汇总。
- `reports/fold_metrics.csv`：每个时间折指标。
- `reports/recommendation_stress_test.csv`：三模型推荐压力测试。
- `reports/safety_summary.csv`：推荐安全门槛结果。
- `reports/optimizer_comparison.csv`：相同范围和预算的 GA/Random Search 结果。
- `reports/champion_summary.json`：最终模型与优化算法。

## 测试

当前 Anaconda 环境的 pytest 输出捕获模块会在测试收集前发生原生崩溃，因此项目使用 Python 内置 unittest：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```
