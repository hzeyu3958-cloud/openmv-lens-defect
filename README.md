# OpenMV 眼镜片缺陷识别 + Android APK 数据显示

本项目是一个 MVP 版本，用传统图像处理和规则分类完成眼镜片表面缺陷检测，并通过蓝牙串口把检测结果发送到 Android 手机显示。它适合课程设计、毕业设计原型演示，后续可以继续升级为机器学习或深度学习分类模型。

## 1. 项目简介

系统分为三端，可按需要使用：

- OpenMV 端：采集镜片图像，检测划痕、灰尘颗粒、污点/油污、镀膜损伤、裂纹、边缘损伤等异常区域，生成 JSON，并通过 UART 发送给蓝牙串口模块。
- Android 端：连接 HC-05/HC-06 等蓝牙串口模块，按行读取 JSON，解析并显示缺陷类型、数量、严重程度、缺陷列表、原始数据和历史记录。
- Windows 上位机端：如果不方便安装手机 APK，可以使用 `windows_host` 里的电脑端上位机。它支持 USB/蓝牙 COM 口接收 JSON，也支持“OpenMV N6 拍图 -> USB 到电脑保存图片 -> 电脑训练模型 -> 模型复制回 N6”的流程。

当前版本不使用复杂深度学习模型，重点是跑通完整流程：

OpenMV 摄像头采集图像  
-> 图像预处理  
-> 缺陷检测  
-> 缺陷分类  
-> JSON 数据发送  
-> Android APK 接收  
-> 手机界面显示

## 2. 项目结构

```text
.
├── openmv/
│   ├── main.py
│   ├── n6_usb_image_capture.py
│   └── n6_classifier_main.py
├── windows_host/
│   ├── lens_defect_host.py
│   ├── run_windows_host.bat
│   ├── build_windows_exe.bat
│   └── README_Windows上位机.md
├── app/
│   ├── build.gradle
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── java/com/example/lensdefect/
│       │   ├── MainActivity.kt
│       │   ├── Defect.kt
│       │   ├── DetectionResult.kt
│       │   └── DefectAdapter.kt
│       └── res/layout/activity_main.xml
├── build.gradle
├── settings.gradle
└── README.md
├── training/
│   └── train_lens_classifier.py
├── dataset/
└── models/
```

## OpenMV N6 训练闭环

如果你选择 OpenMV N6，可以按这个流程做：

```text
OpenMV N6 拍图
-> USB 连接电脑
-> Windows 上位机保存训练图片
-> 电脑训练模型
-> 上位机把 .tflite 和 labels.txt 复制回 OpenMV N6
-> N6 运行模型
-> Windows 上位机显示 JSON 检测结果
```

## 3. OpenMV 使用方法

1. 打开 OpenMV IDE。
2. 连接 OpenMV 摄像头。
3. 把 `openmv/main.py` 复制到 OpenMV。
4. 根据镜片在画面中的位置修改 `LENS_ROI`。
5. 运行脚本，OpenMV IDE 画面会显示 ROI 和缺陷框。
6. 连接蓝牙串口模块后，OpenMV 会通过 UART 每隔 `SEND_INTERVAL_MS` 发送一行 JSON。

OpenMV 主要参数都在 `openmv/main.py` 顶部：

- `LENS_ROI`：镜片检测区域。
- `MIN_CONTRAST_DELTA`：亮度异常阈值。
- `MIN_PIXELS` / `MIN_AREA`：过滤小噪声。
- `SCRATCH_MIN_ASPECT_RATIO`：划痕长宽比阈值。
- `CRACK_MIN_LENGTH`：裂纹长度阈值。
- `UART_PORT` / `UART_BAUDRATE`：串口号和波特率。

N6 训练采图时运行：

```text
openmv/n6_usb_image_capture.py
```

电脑训练完成并复制模型后，N6 运行：

```text
openmv/n6_classifier_main.py
```

## 4. Android APK 使用方法

1. 用 Android Studio 打开本项目根目录。
2. 等待 Gradle 同步完成。
3. 连接 Android 手机，开启开发者模式和 USB 调试。
4. 运行 `app`。
5. 首次启动时允许蓝牙和定位权限。
6. 在系统蓝牙设置中先和 HC-05/HC-06 配对。
7. 回到 APK，点击已配对设备列表中的模块，或选择后点击“连接设备”。
8. 点击“开始接收”，界面会显示 OpenMV 发来的检测结果。
9. 点击“停止接收”可暂停读取，点击“清空记录”会清空历史记录。

## 4.1 Windows 上位机使用方法

如果不想使用手机 APK，可以直接使用 Windows 上位机：

1. Windows 蓝牙设置中先配对 HC-05/HC-06。
2. 查看系统生成的蓝牙串口，例如 `COM5`。
3. 双击 `windows_host/run_windows_host.bat` 运行源码版，或运行 `dist/LensDefectHost.exe`。
4. 选择对应 COM 口，波特率选择 `115200`。
5. 检测时打开“1 检测显示”，点击“开始接收检测 JSON”。
6. 采训练图片时打开“2 采集训练数据”，选择类别后点击“拍一张保存到电脑”。
7. 训练和复制模型时打开“3 训练和部署”。

详细说明见 `windows_host/README_Windows上位机.md`。

## 5. 蓝牙连接方法

OpenMV 与蓝牙串口模块连接：

```text
OpenMV TX  -> 蓝牙模块 RX
OpenMV RX  -> 蓝牙模块 TX
OpenMV GND -> 蓝牙模块 GND
OpenMV VCC -> 蓝牙模块 VCC
```

注意事项：

- TX 和 RX 必须交叉连接。
- GND 必须共地。
- 蓝牙模块供电电压要符合模块要求，常见 HC-05/HC-06 多数小板可接 5V，但请以你的模块标注为准。
- 手机需要先在系统蓝牙设置里和蓝牙模块配对。
- 常见配对密码为 `1234` 或 `0000`。
- OpenMV 端默认波特率为 `115200`，蓝牙模块波特率也要一致。

## 6. JSON 数据格式

OpenMV 每次发送一行 JSON，末尾带换行符 `\n`。示例：

```json
{
  "has_defect": true,
  "defect_count": 2,
  "summary": {
    "scratch": 1,
    "dust": 1,
    "stain": 0,
    "coating_damage": 0,
    "crack": 0,
    "edge_damage": 0,
    "unknown": 0
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

字段说明：

- `has_defect`：是否检测到缺陷。
- `defect_count`：缺陷总数。
- `summary`：每种缺陷数量。
- `overall_level`：整体严重程度，取当前帧中最严重等级。
- `defects`：缺陷详情列表。
- `timestamp`：OpenMV 运行时间戳，单位为毫秒。

## 7. 缺陷类型说明

- `scratch`：划痕，通常细长，长宽比大。
- `dust`：灰尘颗粒，通常面积小，接近点状或圆形。
- `stain`：污点/油污，面积中等，形状不规则。
- `coating_damage`：镀膜损伤，通常是较大片状区域，亮度或纹理异常明显。
- `crack`：裂纹，比普通划痕更长、更明显，严重程度较高。
- `edge_damage`：边缘损伤，出现在 ROI 边缘附近的缺口、暗斑或磨损。
- `unknown`：检测到异常，但规则无法明确分类。

严重程度：

- `normal`：正常
- `light`：轻微
- `medium`：中等
- `serious`：严重

## 8. 参数调试说明

### ROI 怎么调

修改 `openmv/main.py`：

```python
LENS_ROI = (40, 30, 240, 180)
```

四个值分别是 `x, y, w, h`。运行后看 OpenMV IDE 中的矩形框，让它刚好覆盖镜片主体，尽量不要包含夹具和背景。

### 阈值怎么调

主要调：

```python
MIN_CONTRAST_DELTA = 18
STD_FACTOR = 1.35
```

误检太多时增大 `MIN_CONTRAST_DELTA` 或 `STD_FACTOR`；漏检明显划痕时减小它们。

### 面积过滤怎么调

主要调：

```python
MIN_PIXELS = 8
MIN_AREA = 8
DUST_MAX_PIXELS = 45
```

噪声点太多时增大 `MIN_PIXELS` 和 `MIN_AREA`；灰尘识别不到时减小它们。

### 长宽比怎么调

主要调：

```python
SCRATCH_MIN_ASPECT_RATIO = 5.0
SCRATCH_MIN_LENGTH = 24
CRACK_MIN_ASPECT_RATIO = 7.0
CRACK_MIN_LENGTH = 80
```

短污点被误判为划痕时增大长宽比阈值；细划痕漏检时降低长宽比或长度阈值。

### 严重程度怎么调

主要调：

```python
SERIOUS_PIXELS = 900
MEDIUM_PIXELS = 220
COATING_BRIGHTNESS_DELTA = 28
```

如果系统把轻微缺陷判得太严重，增大这些阈值；如果严重缺陷等级偏低，降低这些阈值。

## 9. 后续优化方向

- 增加固定夹具，保证镜片位置一致。
- 增加补光灯，减少环境光变化。
- 使用黑色背景，提高透明镜片边缘和缺陷对比度。
- 使用暗场照明，让划痕更明显。
- 增加拍照保存功能。
- 增加检测报告导出功能。
- 收集图片数据集。
- 后期训练机器学习分类模型。
- 后期支持更多缺陷类型。

## 10. 演示建议

演示时建议准备几片样品：正常镜片、带明显划痕的镜片、带灰尘颗粒的镜片、边缘破损或贴纸模拟污点的镜片。先固定镜片和光源，再调 `LENS_ROI` 和阈值，能显著提高演示稳定性。
