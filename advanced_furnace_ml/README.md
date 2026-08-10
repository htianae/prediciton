# Advanced Furnace ML

这是与现有生产项目隔离的高级离线实验。它使用292炉 Excel 数据比较直接总气耗和单位气耗路线，以及 Ridge、ElasticNet、Huber、GAM、GPR、CatBoost、LightGBM、两种残差混合模型和 OOF 集成。

## 验证设计

- 前248炉用于开发；
- 三个时间折为148/33、181/33、214/34；
- 最后44炉锁定，只在模型与参数冻结后审计；
- 最终模型使用全部292炉重新训练。

## 正式训练

```bash
cd /Users/tian/Desktop/prediction_project
PYTHONPATH=advanced_furnace_ml/src python advanced_furnace_ml/train_advanced.py \
  --excel 4_month_data_2026_02_01_2026_06_25.xlsx \
  --output-dir advanced_furnace_ml \
  --optimizer-budget 600 \
  --seeds 0 1 2
```

## 推荐通过条件

模型推荐只有同时满足下列条件才标记为 A 级离线试验候选：

- 保守预测气耗低于86000 kg相似历史炉次实际气耗中位数；
- 至少2/3大窗口时间折预测能够节气；
- 历史近邻距离合格；
- 不命中搜索边界；
- 位于历史低气耗参数附近±10%信任区域。

没有候选通过时返回历史相似低气耗方案，不强行制造“安全推荐”。离线预计节省不等于工厂实际节省，必须经过现场受控试验。

优化器使用种子0、1、2比较稳定性；选定优化器后，正式推荐固定使用部署种子42。因此报告、joblib模型包和命令行在输入及预算相同时会返回相同结果。

## 加载模型包

`advanced_furnace_bundle.joblib` 不只是模型参数，还包含特征顺序、预处理、三个时间折模型、90%预测区间、安全门所需的历史参考和推荐模型。因为模型包引用本项目中的自定义类，加载时必须让 Python 能找到 `src`：

```bash
cd /Users/tian/Desktop/prediction_project
PYTHONPATH=advanced_furnace_ml/src python advanced_furnace_ml/predict_advanced.py \
  --model advanced_furnace_ml/artifacts/advanced_furnace_bundle.joblib \
  --total-weight 86000 --solid-ratio 38 --melting-time 8 \
  --waiting-time 0.65 --door-open-count 14 --door-open-duration 69.4
```

推荐命令：

```bash
python advanced_furnace_ml/recommend_advanced.py \
  --model advanced_furnace_ml/artifacts/advanced_furnace_bundle.joblib \
  --total-weight 86000 --budget 600 --seed 42
```

## 未来实时数据

`contracts/` 允许工人漏填、延迟填写和不按整点更新。当前没有历史过程时间序列，因此项目不会伪造实时剩余气耗模型结果。
