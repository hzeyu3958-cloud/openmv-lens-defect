# Windows 上位机使用说明

## 1. 功能

这个上位机现在支持完整流程：

```text
OpenMV N6 拍图
-> USB 连接电脑
-> 电脑保存训练图片
-> 电脑训练模型
-> 把模型文件传回 OpenMV N6
-> N6 运行模型并输出 JSON
-> Windows 上位机显示检测结果
```

上位机功能包括：

- 串口连接状态
- 是否检测到缺陷
- 缺陷总数
- 整体严重程度
- 各类缺陷数量
- 缺陷详情列表
- 最近一次 JSON 原始数据
- 历史检测记录
- 历史记录 CSV 导出
- 按类别采集训练图片并保存到电脑
- 启动电脑端训练脚本
- 把 `.tflite` 模型和标签文件复制回 OpenMV N6

## 2. 推荐 USB 连接方式

如果你不用手机 APK，也不想用蓝牙，推荐：

```text
OpenMV N6 -> USB 线 -> Windows 电脑
```

Windows 会识别出一个 USB 串口 COM 口，上位机直接选择这个 COM 口即可。

## 3. 蓝牙连接方式，可选

OpenMV 与蓝牙串口模块：

```text
OpenMV TX  -> 蓝牙模块 RX
OpenMV RX  -> 蓝牙模块 TX
OpenMV GND -> 蓝牙模块 GND
OpenMV VCC -> 蓝牙模块 VCC
```

注意：

- TX 和 RX 要交叉连接。
- GND 必须共地。
- 蓝牙模块供电电压要符合模块要求。
- 常见模块为 HC-05 或 HC-06。
- Windows 需要先在系统蓝牙设置中配对蓝牙模块。
- 常见配对密码为 `1234` 或 `0000`。

## 4. Windows COM 口

USB 或蓝牙都会在 Windows 上形成 COM 口。查看方式：

```text
控制面板 -> 设备和打印机 -> 更多蓝牙设置 -> COM 端口
```

也可以在设备管理器中查看：

```text
端口 (COM 和 LPT)
```

USB 连接 OpenMV N6 时，一般会看到 OpenMV 相关串口；蓝牙模块则类似：

```text
Standard Serial over Bluetooth link (COM5)
```

上位机中选择对应 COM 口，波特率默认 `115200`。

## 5. 采集训练图片

1. 在 OpenMV IDE 中打开并运行：

```text
openmv/n6_usb_image_capture.py
```

2. 上位机打开“2 采集训练数据”页。
3. 选择数据集目录，点击“初始化分类文件夹”。
4. 选择类别，例如 `scratch`。
5. 点击“拍一张保存到电脑”。
6. 图片会保存到：

```text
dataset/train/scratch/
```

也可以选择 `val` 或 `test`，用于验证集和测试集。

建议每类至少先采 100 张，效果更稳建议每类 300 张以上。

## 6. 电脑训练模型

上位机打开“3 训练和部署”页，点击“启动本地训练脚本”。

训练脚本位置：

```text
training/train_lens_classifier.py
```

如果电脑没有 TensorFlow，需要先安装：

```text
pip install -r training/requirements-training.txt
```

训练输出目录：

```text
models/
```

训练完成后会生成：

```text
models/lens_defect_classifier_int8.tflite
models/lens_defect_labels.txt
```

## 7. 把模型传回 OpenMV N6

1. 用 USB 连接 OpenMV N6。
2. Windows 文件管理器里打开 OpenMV 的 USB 盘符。
3. 在上位机“3 训练和部署”页选择：
   - 模型：`models/lens_defect_classifier_int8.tflite`
   - 标签：`models/lens_defect_labels.txt`
   - OpenMV 盘符/目录
4. 点击“复制模型到 OpenMV N6”。

复制后 OpenMV 根目录应有：

```text
 lens_defect_classifier_int8.tflite
 lens_defect_labels.txt
```

## 8. N6 运行模型

在 OpenMV IDE 中运行：

```text
openmv/n6_classifier_main.py
```

然后上位机打开“1 检测显示”页，点击“开始接收检测 JSON”，即可显示 N6 模型输出结果。

## 9. 运行源码版

双击：

```text
windows_host/run_windows_host.bat
```

第一次运行会自动创建 `.venv_windows_host` 虚拟环境并安装 `pyserial`。

## 10. 运行 exe 版

如果已经打包成功，直接双击：

```text
dist/LensDefectHost.exe
```

## 11. 打包 exe

双击：

```text
windows_host/build_windows_exe.bat
```

成功后 exe 在：

```text
dist/LensDefectHost.exe
```

## 12. 检测显示使用步骤

1. 给 OpenMV N6 上电并运行检测脚本。
2. USB 连接电脑，或使用蓝牙模块。
3. 运行 `LensDefectHost.exe`。
4. 点击“刷新串口”。
5. 选择 OpenMV 或蓝牙串口对应的 COM 口。
6. 波特率选择 `115200`。
7. 点击“开始接收”。
8. OpenMV 每发送一行 JSON，上位机会刷新检测结果。

如果没有硬件，也可以点击“模拟一条”测试界面显示效果。

## 13. 常见问题

### 找不到 COM 口

请确认蓝牙模块已经在 Windows 蓝牙设置中完成配对，并查看设备管理器里是否出现 COM 端口。

### 打开串口失败

同一个 COM 口只能被一个软件占用。请关闭串口调试助手、OpenMV IDE 的串口监视器或其他占用该 COM 口的软件。

### 收不到数据

检查：

- OpenMV 的 `UART_PORT` 是否接对。
- OpenMV 的 `UART_BAUDRATE` 是否和上位机一致。
- TX/RX 是否交叉连接。
- GND 是否共地。
- OpenMV 发送的 JSON 是否以 `\n` 结尾。

### JSON 解析失败

上位机要求 OpenMV 每条数据是一整行 JSON。请确认 `openmv/main.py` 中发送格式为：

```python
uart.write(line + "\n")
```

## 14. 数据闭环优化说明

新版上位机在“2 采集训练数据”页增加了训练数据闭环能力：

- 拍照后显示最近一张图片预览。
- 自动检查亮度、对比度和尺寸，并在质量提示里标出偏暗、过曝、低对比度等问题。
- 默认按 `train 70% / val 20% / test 10%` 自动分配样本，减少手工搬图。
- 每次保存图片时自动追加 `dataset/metadata.csv`，记录文件名、类别、集合、尺寸、亮度、对比度、串口和采集时间。
- 数据集统计区会提示类别数量不足、缺少验证/测试集、类别不均衡等风险。

“3 训练和部署”页会读取训练脚本生成的 `models/training_summary.json`，显示最近一次训练的 accuracy、loss、INT8 模型路径和混淆矩阵文件路径。
