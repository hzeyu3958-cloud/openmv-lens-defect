# 模型输出目录

电脑训练脚本会把模型输出到这里：

- `lens_defect_classifier.keras`
- `lens_defect_classifier_float.tflite`
- `lens_defect_classifier_int8.tflite`
- `lens_defect_labels.txt`
- `training_summary.json`
- `training_history.csv`
- `confusion_matrix_val.csv` 或 `confusion_matrix_test.csv`

Windows 上位机“训练和部署”页可以把 `.tflite` 和 `labels.txt` 复制到 OpenMV N6。

其中 `training_summary.json` 会被上位机读取，用来显示最近一次训练的 accuracy、loss、类别顺序和模型文件路径。混淆矩阵 CSV 用来检查哪些缺陷类型容易互相误判。
