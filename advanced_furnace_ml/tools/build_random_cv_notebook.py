"""Build the self-contained random holdout/CV prediction notebook."""

from pathlib import Path
import pprint

import nbformat as nbf


PROJECT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT / "advanced_furnace_ml/src/advanced_furnace_ml"
DESTINATION = PROJECT / "advanced_furnace_ml/notebooks/random_cv_model_comparison.ipynb"
RUNTIME_FILES = {
    "__init__.py", "data.py", "experiment.py", "features.py",
    "models.py", "uncertainty.py", "validation.py",
}


def markdown(title: str, body: str = ""):
    return nbf.v4.new_markdown_cell(f"## {title}\n\n{body}".strip())


def runtime_sources() -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(SOURCE_DIR.glob("*.py"))
        if path.name in RUNTIME_FILES
    }


def embedded_runtime_cell() -> str:
    rendered = pprint.pformat(runtime_sources(), width=120, sort_dicts=True)
    return (
        "# 执行时只使用下面内嵌的源代码快照，不读取项目Python模块。\n"
        f"RUNTIME_SOURCES = {rendered}\n\n"
        "import importlib\n"
        "import sys\n"
        "RUNTIME_ROOT = OUTPUT_ROOT / 'runtime'\n"
        "PACKAGE_DIR = RUNTIME_ROOT / 'random_cv_runtime'\n"
        "PACKAGE_DIR.mkdir(parents=True, exist_ok=True)\n"
        "for filename, source_text in RUNTIME_SOURCES.items():\n"
        "    (PACKAGE_DIR / filename).write_text(source_text, encoding='utf-8')\n"
        "if str(RUNTIME_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(RUNTIME_ROOT))\n"
        "importlib.invalidate_caches()\n"
        "print(f'独立运行时已生成: {PACKAGE_DIR}')"
    )


def build_notebook(destination: Path = DESTINATION) -> Path:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# 随机80/20留出与训练集5折CV模型比较\n\n"
            "保留全部292炉，只比较预测模型；测试集不参与模型选择。"
        ),
        markdown("配置、依赖与自包含运行时"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import importlib.util\n"
            "required = ['numpy','pandas','scipy','sklearn','joblib','openpyxl','lightgbm','catboost']\n"
            "missing = [name for name in required if importlib.util.find_spec(name) is None]\n"
            "if missing:\n"
            "    raise ImportError(f'缺少依赖: {missing}')\n"
            "EXCEL_NAME = '4_month_data_2026_02_01_2026_06_25.xlsx'\n"
            "EXCEL_PATH = next((root / EXCEL_NAME for root in [Path.cwd(), *Path.cwd().parents] if (root / EXCEL_NAME).exists()), None)\n"
            "if EXCEL_PATH is None:\n"
            "    raise FileNotFoundError(EXCEL_NAME)\n"
            "PROJECT_ROOT = EXCEL_PATH.parent\n"
            "OUTPUT_ROOT = PROJECT_ROOT / 'advanced_furnace_ml/random_cv_output'\n"
            "ARTIFACTS_DIR = OUTPUT_ROOT / 'artifacts'\n"
            "REPORTS_DIR = OUTPUT_ROOT / 'reports'\n"
            "for directory in (OUTPUT_ROOT, ARTIFACTS_DIR, REPORTS_DIR):\n"
            "    directory.mkdir(parents=True, exist_ok=True)\n"
            "print({'excel': str(EXCEL_PATH), 'output': str(OUTPUT_ROOT)})"
        ),
        nbf.v4.new_code_cell(embedded_runtime_cell()),
        markdown("Excel数据检查：保留全部292炉"),
        nbf.v4.new_code_cell(
            "import numpy as np\n"
            "import pandas as pd\n"
            "from random_cv_runtime.data import FEATURE_COLS, TARGET_COL, load_batch_data, mark_target_outliers\n"
            "df = mark_target_outliers(load_batch_data(EXCEL_PATH))\n"
            "assert len(df) == 292\n"
            "outlier_count_before_split = int(df['is_high_gas_outlier'].sum())\n"
            "display(df.head())\n"
            "display({'总炉次': len(df), '异常炉次标记数（保留）': outlier_count_before_split, '删除炉次': 0})"
        ),
        markdown("随机训练集与测试集"),
        nbf.v4.new_code_cell(
            "from sklearn.model_selection import KFold, train_test_split\n"
            "train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)\n"
            "train_df = train_df.reset_index(drop=True)\n"
            "test_df = test_df.reset_index(drop=True)\n"
            "assert len(train_df) == 233\n"
            "assert len(test_df) == 59\n"
            "outlier_count_after_split = int(train_df['is_high_gas_outlier'].sum() + test_df['is_high_gas_outlier'].sum())\n"
            "assert outlier_count_after_split == outlier_count_before_split\n"
            "kfold = KFold(n_splits=5, shuffle=True, random_state=42)\n"
            "random_splits = list(kfold.split(train_df))\n"
            "assert len(random_splits) == 5\n"
            "for train_index, valid_index in random_splits:\n"
            "    assert set(train_index).isdisjoint(set(valid_index))\n"
            "    assert int(max(train_index.max(), valid_index.max())) < len(train_df)\n"
            "split_table = pd.DataFrame([\n"
            "    {'fold': i + 1, 'train_size': len(tr), 'valid_size': len(va), 'overlap': len(set(tr) & set(va))}\n"
            "    for i, (tr, va) in enumerate(random_splits)\n"
            "])\n"
            "display({'训练炉次': len(train_df), '测试炉次': len(test_df), 'random_state': 42})\n"
            "display(split_table)"
        ),
        markdown("训练集内部5折交叉验证与19候选"),
        nbf.v4.new_markdown_cell(
            "LightGBM/CatBoost小数据参数、OOF权重和Champion都只使用233炉训练集的五折结果。"
        ),
        nbf.v4.new_code_cell(
            "from random_cv_runtime.experiment import run_model_matrix\n"
            "random_experiment = run_model_matrix(train_df, random_splits)\n"
            "random_cv_summary = random_experiment.summary.copy().sort_values(\n"
            "    ['selection_score', 'mae_mean', 'candidate']\n"
            ").reset_index(drop=True)\n"
            "selected_before_test = str(random_cv_summary.iloc[0]['candidate'])\n"
            "random_experiment.selected_name = selected_before_test\n"
            "random_cv_summary['selected_before_test'] = random_cv_summary['candidate'].eq(selected_before_test)\n"
            "random_cv_fold_metrics = pd.concat(\n"
            "    [result.fold_metrics for result in random_experiment.results.values()], ignore_index=True\n"
            ")\n"
            "assert random_cv_summary['candidate'].nunique() == 19\n"
            "assert len(random_experiment.ensemble_weights) == 4\n"
            "assert np.all(random_experiment.ensemble_weights >= 0)\n"
            "assert abs(float(random_experiment.ensemble_weights.sum()) - 1.0) < 1e-10\n"
            "ensemble_table = pd.DataFrame({\n"
            "    'member': random_experiment.ensemble_members, 'weight': random_experiment.ensemble_weights\n"
            "})\n"
            "random_experiment.freeze()\n"
            "display(random_cv_summary.round(4))\n"
            "display(random_experiment.tree_tuning_summary.round(4))\n"
            "display(ensemble_table.round(4))\n"
            "print(f'CV Champion（测试集尚未查看）: {selected_before_test}')"
        ),
        markdown("冻结Champion后的59炉测试审计"),
        nbf.v4.new_code_cell(
            "from random_cv_runtime.experiment import audit_frozen_models\n"
            "random_test_audit = audit_frozen_models(random_experiment, test_df)\n"
            "assert len(random_test_audit) == 19\n"
            "required_test_columns = {\n"
            "    'mae', 'rmse', 'wape', 'r2', 'error_gt_10pct_rate',\n"
            "    'rmse_bootstrap_low95', 'rmse_bootstrap_high95',\n"
            "    'interval_coverage_90', 'interval_mean_width',\n"
            "}\n"
            "assert required_test_columns.issubset(random_test_audit.columns)\n"
            "test_rmse_winner = str(random_test_audit.sort_values('rmse').iloc[0]['candidate'])\n"
            "interpretation = pd.DataFrame([\n"
            "    {'role': 'CV Champion', 'candidate': selected_before_test, 'meaning': '训练集内部5折CV选出'},\n"
            "    {'role': 'test RMSE winner', 'candidate': test_rmse_winner, 'meaning': '59炉测试集审计优胜者，不反向改写选模'},\n"
            "])\n"
            "display(random_test_audit.sort_values('rmse').round(4))\n"
            "display(interpretation)\n"
            "print(f'CV Champion: {selected_before_test}')\n"
            "print(f'test RMSE winner: {test_rmse_winner}')\n"
            "print('测试集结果不能反向改变CV Champion。')"
        ),
        markdown("随机CV与时间滚动CV比较"),
        nbf.v4.new_code_cell(
            "from random_cv_runtime.validation import large_window_splits\n"
            "chron_dev = df.iloc[:248].copy()\n"
            "chronological_experiment = run_model_matrix(chron_dev, large_window_splits(248))\n"
            "chronological_cv_summary = chronological_experiment.summary.copy()\n"
            "metric_columns = ['candidate', 'rmse_mean', 'rmse_std', 'mae_mean', 'selection_score']\n"
            "random_for_merge = random_cv_summary[metric_columns].rename(columns={\n"
            "    column: f'random_{column}' for column in metric_columns if column != 'candidate'\n"
            "})\n"
            "chron_for_merge = chronological_cv_summary[metric_columns].rename(columns={\n"
            "    column: f'chronological_{column}' for column in metric_columns if column != 'candidate'\n"
            "})\n"
            "random_vs_chronological = random_for_merge.merge(chron_for_merge, on='candidate', how='inner')\n"
            "random_vs_chronological['rmse_difference_random_minus_chronological'] = (\n"
            "    random_vs_chronological['random_rmse_mean']\n"
            "    - random_vs_chronological['chronological_rmse_mean']\n"
            ")\n"
            "assert len(random_vs_chronological) == 19\n"
            "display(random_vs_chronological.sort_values('random_selection_score').round(4))\n"
            "print('随机划分会混合早期和晚期炉次，因此随机CV结果可能对未来炉次偏乐观。')"
        ),
        markdown("最终单模型重训与joblib"),
        nbf.v4.new_code_cell(
            "import json\n"
            "import warnings\n"
            "from datetime import datetime, timezone\n"
            "import joblib\n"
            "from sklearn.base import clone\n"
            "from sklearn.exceptions import ConvergenceWarning\n"
            "champion_result = random_experiment.results[selected_before_test]\n"
            "final_champion = clone(champion_result.estimator_template)\n"
            "with warnings.catch_warnings():\n"
            "    warnings.simplefilter('ignore', ConvergenceWarning)\n"
            "    final_champion.fit(df[FEATURE_COLS], df[TARGET_COL])\n"
            "artifact_path = ARTIFACTS_DIR / 'random_cv_champion.joblib'\n"
            "bundle = {\n"
            "    'artifact_version': 'random-cv-furnace-1.0',\n"
            "    'created_at_utc': datetime.now(timezone.utc).isoformat(),\n"
            "    'feature_cols': list(FEATURE_COLS),\n"
            "    'target_col': TARGET_COL,\n"
            "    'training_batches': int(len(df)),\n"
            "    'outlier_batches_retained': outlier_count_before_split,\n"
            "    'selected_model_name': selected_before_test,\n"
            "    'model': final_champion,\n"
            "    'cv_fold_models': champion_result.fold_models,\n"
            "    'conformal_radius_90': float(champion_result.conformal_radius_90),\n"
            "    'split_config': {\n"
            "        'test_size': 0.20, 'random_state': 42, 'cv_folds': 5,\n"
            "        'cv_shuffle': True, 'all_rows_final_refit': True,\n"
            "    },\n"
            "}\n"
            "joblib.dump(bundle, artifact_path)\n"
            "random_cv_summary.to_csv(REPORTS_DIR / 'random_cv_summary.csv', index=False)\n"
            "random_cv_fold_metrics.to_csv(REPORTS_DIR / 'random_cv_fold_metrics.csv', index=False)\n"
            "random_test_audit.to_csv(REPORTS_DIR / 'random_test_audit.csv', index=False)\n"
            "chronological_cv_summary.to_csv(REPORTS_DIR / 'chronological_cv_summary.csv', index=False)\n"
            "random_vs_chronological.to_csv(REPORTS_DIR / 'random_vs_chronological.csv', index=False)\n"
            "random_experiment.tree_tuning_summary.to_csv(REPORTS_DIR / 'tree_tuning_summary.csv', index=False)\n"
            "ensemble_table.to_csv(REPORTS_DIR / 'ensemble_weights.csv', index=False)\n"
            "run_summary = {\n"
            "    'artifact_version': bundle['artifact_version'],\n"
            "    'selected_model_name': selected_before_test,\n"
            "    'test_rmse_winner': test_rmse_winner,\n"
            "    'training_batches': int(len(df)),\n"
            "    'train_batches_for_selection': int(len(train_df)),\n"
            "    'test_batches_for_audit': int(len(test_df)),\n"
            "    'candidate_count': int(random_cv_summary['candidate'].nunique()),\n"
            "    'artifact_path': str(artifact_path),\n"
            "}\n"
            "(REPORTS_DIR / 'run_summary.json').write_text(\n"
            "    json.dumps(run_summary, ensure_ascii=False, indent=2), encoding='utf-8'\n"
            ")\n"
            "print(f'Champion artifact: {artifact_path}')\n"
            "display(run_summary)"
        ),
        nbf.v4.new_code_cell(
            "def validate_frame(input_frame, expected_columns):\n"
            "    if not isinstance(input_frame, pd.DataFrame):\n"
            "        raise TypeError('输入必须是pandas DataFrame。')\n"
            "    expected = list(expected_columns)\n"
            "    missing = [column for column in expected if column not in input_frame.columns]\n"
            "    extra = [column for column in input_frame.columns if column not in expected]\n"
            "    if missing:\n"
            "        raise ValueError(f'缺少特征列: {missing}')\n"
            "    if extra:\n"
            "        raise ValueError(f'存在未声明特征列: {extra}')\n"
            "    numeric = input_frame[expected].apply(pd.to_numeric, errors='coerce')\n"
            "    if numeric.isna().any().any():\n"
            "        bad = numeric.columns[numeric.isna().any()].tolist()\n"
            "        raise ValueError(f'特征包含非数值或缺失值: {bad}')\n"
            "    return numeric\n"
            "\n"
            "def predict_bundle(loaded_bundle, input_frame):\n"
            "    numeric = validate_frame(input_frame, loaded_bundle['feature_cols'])\n"
            "    prediction = np.asarray(loaded_bundle['model'].predict(numeric), dtype=float)\n"
            "    radius = float(loaded_bundle['conformal_radius_90'])\n"
            "    fold_matrix = np.vstack([\n"
            "        model.predict(numeric) for model in loaded_bundle['cv_fold_models']\n"
            "    ])\n"
            "    return pd.DataFrame({\n"
            "        'predicted_total_gas': prediction,\n"
            "        'interval_90_low': np.maximum(0.0, prediction - radius),\n"
            "        'interval_90_high': prediction + radius,\n"
            "        'cv_fold_prediction_std': fold_matrix.std(axis=0),\n"
            "    }, index=input_frame.index)"
        ),
        markdown("最终自动测试汇总"),
        nbf.v4.new_code_cell(
            "loaded_bundle = joblib.load(artifact_path)\n"
            "replay_input = df[FEATURE_COLS].iloc[:3].copy()\n"
            "before_save = predict_bundle(bundle, replay_input)\n"
            "after_load = predict_bundle(loaded_bundle, replay_input)\n"
            "np.testing.assert_allclose(\n"
            "    before_save.to_numpy(), after_load.to_numpy(), rtol=0.0, atol=1e-10\n"
            ")\n"
            "prediction_round_trip_atol_1e_10 = True\n"
            "\n"
            "def rejects(frame):\n"
            "    try:\n"
            "        predict_bundle(loaded_bundle, frame)\n"
            "    except (TypeError, ValueError):\n"
            "        return True\n"
            "    return False\n"
            "\n"
            "missing_column_rejected = rejects(replay_input.drop(columns=[FEATURE_COLS[0]]))\n"
            "extra_column_rejected = rejects(replay_input.assign(undeclared_feature=1.0))\n"
            "bad_numeric = replay_input.copy()\n"
            "bad_numeric.loc[bad_numeric.index[0], FEATURE_COLS[0]] = 'not-a-number'\n"
            "nonnumeric_value_rejected = rejects(bad_numeric)\n"
            "artifact_replay_tests = pd.DataFrame([\n"
            "    {'test': 'prediction_round_trip_atol_1e_10', 'passed': prediction_round_trip_atol_1e_10},\n"
            "    {'test': 'missing_column_rejected', 'passed': missing_column_rejected},\n"
            "    {'test': 'extra_column_rejected', 'passed': extra_column_rejected},\n"
            "    {'test': 'nonnumeric_value_rejected', 'passed': nonnumeric_value_rejected},\n"
            "    {'test': 'all_292_batches_retained', 'passed': len(df) == 292},\n"
            "    {'test': 'nineteen_candidates_compared', 'passed': len(random_cv_summary) == 19},\n"
            "])\n"
            "assert artifact_replay_tests['passed'].all(), artifact_replay_tests\n"
            "display(artifact_replay_tests)\n"
            "display(after_load.round(4))\n"
            "print('全部自动测试通过；joblib只提供气耗预测，不包含参数推荐。')"
        ),
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, destination)
    return destination


if __name__ == "__main__":
    print(build_notebook())
