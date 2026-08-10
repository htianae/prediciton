# 独立三模型选择与参数推荐 Notebook 设计

## 目标

创建一份真正可以独立运行的 Jupyter Notebook。用户只需要：

1. 本 notebook；
2. `4_month_data_2026_02_01_2026_06_25.xlsx`；
3. 已安装 Python、pandas、NumPy、scikit-learn、LightGBM、matplotlib、joblib。

Notebook 不导入 `furnace_champion`，其内部包含测试阶段的全部数据处理、模型评估、优化推荐和安全检查代码。

## 文件

- 源文件：`Nanshang_three_model_standalone.ipynb`
- 已执行文件：`Nanshang_three_model_standalone_executed.ipynb`

现有模块化 notebook、生产模型包、API 示例和正式报告均不覆盖。

## Notebook 内容

### 1. 配置

集中设置：

- Excel 路径；
- 推荐重量 `86000.0 kg`；
- 5 折时间滚动验证；
- GA/随机搜索单次预算 `5000`；
- 比较种子 `0–9`；
- 最终推荐种子 `42`；
- `EXPORT_JOBLIB = False`，需要时可切换为 `True`。

### 2. 完整数据代码

Notebook 内部实现 Excel 转置、字段映射、数值化、完整炉次筛选和高气耗 IQR 标记。高气耗炉次保留，不删除。Excel 炉次列顺序作为时间顺序。

### 3. 完整模型代码

Notebook 内部定义：

- LightGBM pipeline；
- Linear pipeline；
- Huber pipeline；
- 五折 expanding-window 时间验证；
- MAE、RMSE、R²；
- `平均RMSE + 0.25 × RMSE标准差` 选择分数；
- Champion 排名规则；
- 使用全部历史数据重新训练三个模型。

### 4. 完整推荐代码

Notebook 内部定义：

- 86000 kg 相似炉次筛选；
- P5–P95 统一搜索空间；
- RobustScaler + kNN 历史可行性；
- 跨折模型预测标准差；
- `预测气耗 + 不确定性 + 历史距离惩罚` 目标；
- 随机搜索；
- 遗传算法；
- 相同预算、范围和随机种子比较；
- 前50个可行候选中位数推荐；
- 历史相似低气耗炉次兜底。

开门次数为整数，并严格满足数值 P5–P95 边界，不使用会越界的向外取整。

### 5. 结果展示

Notebook 输出：

1. 数据质量表；
2. 三模型时间验证汇总；
3. 每折 RMSE 表与折线图；
4. 86000 kg 推荐上下文和搜索边界；
5. 三模型 × 两优化器汇总；
6. 三个模型原始推荐参数；
7. 每个推荐的预测气耗、不确定性、历史距离和边界警告；
8. 历史低气耗兜底；
9. 经过安全门后的生产候选；
10. 自动结论。

模型推荐未通过安全门时仍保留在“原始推荐”表中，仅在“生产候选”表中切换到历史兜底。

### 6. 可选模型导出

`EXPORT_JOBLIB = False` 时不写任何模型文件。设置为 `True` 时，在 notebook 末尾将以下内容保存为一个 joblib bundle：

- 三个全量训练模型；
- 三组时间折模型；
- 特征顺序和目标字段；
- 模型对比指标；
- 86000 kg 原始推荐和生产候选；
- Excel 训练炉次数；
- 生成时间。

导出文件使用独立名称 `artifacts/three_model_standalone.joblib`，不覆盖 `gas_champion.joblib`。

## 验收标准

1. Notebook 源码中不存在 `from furnace_champion` 或 `import furnace_champion`。
2. 将 notebook 和 Excel 单独放在同一目录即可执行。
3. 所有代码单元从头执行完成，无错误输出。
4. 输出恰好包含 Linear、Huber、LightGBM 三个模型。
5. 三模型均产生一行原始推荐和一行安全处理后的生产候选。
6. GA 与随机搜索严格共享范围、预算和种子。
7. 所有候选参数均位于展示的搜索边界内。
8. 默认执行不覆盖任何现有 joblib、API 文件或报告。
