# OpenMV 眼镜片缺陷识别上位机

本项目用于眼镜片表面缺陷检测，当前只保留 Windows 上位机、OpenMV/N6 脚本、电脑端训练和部署流程。检测类别固定为：

```text
正常 normal
划痕 scratch
灰尘颗粒 dust
污点/油污 stain
```

## 1. 系统流程

```text
OpenMV/N6 拍照
-> 一级规则算法快速找可疑缺陷
-> USB 串口把 JSON 和画面发给 Windows 上位机
-> Windows 上位机显示实时画面并自动红圈标注
-> 可选：电脑端二级异常检测复核
-> 可选：OpenMV UART 输出结果给单片机
```

传送带场景建议先采用“到位暂停拍照”方式：

```text
到位传感器
-> 单片机暂停传送带
-> OpenMV 拍照识别
-> UART 输出 normal / scratch / dust / stain
-> 单片机继续运行或分拣
```

## 2. 项目结构

```text
.
├── openmv/
│   ├── main.py
│   ├── n6_usb_image_capture.py
│   ├── n6_lens_tracker_main.py
│   ├── n6_lens_defect_rule_main.py
│   └── n6_classifier_main.py
├── windows_host/
│   ├── lens_defect_host.py
│   ├── pc_camera_rule_test.py
│   ├── stage2_anomaly.py
│   ├── run_windows_host.bat
│   ├── run_pc_camera_test.bat
│   ├── run_pc_camera_two_stage_test.bat
│   ├── build_windows_exe.bat
│   └── README_Windows上位机.md
├── training/
│   ├── train_lens_classifier.py
│   ├── train_stage2_anomaly.py
│   └── train_stage2_anomaly.bat
├── dataset/
├── models/
└── release/
    └── LensDefectHost.exe
```

## 3. Windows 上位机

源码版运行：

```bat
windows_host\run_windows_host.bat
```

打包版运行：

```bat
release\LensDefectHost.exe
```

主要功能：

- 接收 OpenMV/N6 输出的 JSON。
- 显示 OpenMV 实时画面。
- 自动用红圈标出划痕、灰尘颗粒、污点。
- 采集训练图片并按 `train/val/test` 保存。
- 启动本地训练脚本。
- 复制模型到 OpenMV/N6。
- 可选启用电脑端二级异常检测复核。

## 4. OpenMV/N6 脚本

采集训练图片时运行：

```text
openmv/n6_usb_image_capture.py
```

没有训练模型时，先运行规则检测版：

```text
openmv/n6_lens_defect_rule_main.py
```

只调镜片跟踪时运行：

```text
openmv/n6_lens_tracker_main.py
```

训练模型复制到 N6 后运行：

```text
openmv/n6_classifier_main.py
```

需要把识别结果给单片机时，在 OpenMV 脚本里打开 UART 输出。N6 的 UART3 默认引脚为：

```text
P4 = TX
P5 = RX
GND 必须共地
信号电平按 3.3V 处理
```

## 5. 训练数据

上位机采集数据时，目录结构为：

```text
dataset/
├── train/
│   ├── normal/
│   ├── scratch/
│   ├── dust/
│   └── stain/
├── val/
└── test/
```

建议先采集：

```text
normal 正常：100 张以上
scratch 划痕：100 张以上
dust 灰尘颗粒：100 张以上
stain 污点/油污：100 张以上
```

效果更稳时，每类建议 300 张以上。

## 6. 训练分类模型

上位机“3 训练和部署”页可以启动训练。也可以手动运行：

```bat
.venv_training\Scripts\python.exe training\train_lens_classifier.py --dataset dataset --output models --epochs 30 --batch-size 8 --image-size 128 --model tiny_depthwise
```

训练完成后会生成：

```text
models/lens_defect_classifier_int8.tflite
models/lens_defect_labels.txt
```

默认 `tiny_depthwise` 是给 OpenMV/N6 准备的轻量模型，参数量比普通 CNN 更小，导出的 `int8.tflite` 更适合放到 N6 上运行。需要对比旧模型时可以改成：

```bat
--model tiny_cnn
```

## 7. 二级异常检测

电脑端二级异常检测用于减少误检，并补充 OpenMV 规则算法没有抓到的异常区域。它需要先采集正常镜片图片，然后训练正常参考模型：

```bat
training\train_stage2_anomaly.bat
```

输出文件：

```text
models/lens_stage2_anomaly.npz
```

两级检测电脑摄像头测试：

```bat
windows_host\run_pc_camera_two_stage_test.bat
```

Windows 上位机检测页也可以直接启用“电脑二级复核”。

## 8. JSON 数据格式

OpenMV 每次发送一行 JSON，末尾带换行符 `\n`。示例：

```json
{
  "has_defect": true,
  "defect_count": 2,
  "summary": {
    "scratch": 1,
    "dust": 1,
    "stain": 0
  },
  "overall_level": "medium",
  "defects": [
    {
      "type": "scratch",
      "confidence": 0.86,
      "x": 120,
      "y": 80,
      "w": 70,
      "h": 5,
      "area": 350,
      "length": 70,
      "aspect_ratio": 14.0,
      "level": "medium"
    }
  ],
  "timestamp": 123456
}
```

## 9. 调试建议

- 固定镜片位置，尽量让镜片每次进入同一画面区域。
- 使用纯色背景，避免镜片后面有复杂纹理。
- 使用侧向 LED 或暗场照明，让划痕更明显。
- 传送带场景优先暂停拍照，先保证清晰度。
- 误检多时先检查光源和镜片 mask，再调阈值。
- 训练数据不要混类，正常样本必须是真正干净的正常镜片。
