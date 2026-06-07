# Windows 上位机说明

当前上位机只保留一个程序：

```text
release/LensDefectHost.exe
```

## 1. 通讯方式

上位机现在有两路通讯：

- OpenMV N6：仍然使用 USB 串口通讯，负责接收识别 JSON 和实时画面。
- 蓝牙模块：在上位机里点击“扫描蓝牙”，选择扫描到的蓝牙设备名，然后连接模块发送识别结果。

## 2. 蓝牙发送内容

蓝牙模块上电后，先在上位机点击“扫描蓝牙”，选择类似 `YANGYANG`、`BH207-1` 这样的设备名，再点击“连接蓝牙模块”。蓝牙连接界面不需要选择波特率，上位机会用内部默认值连接。

连接成功后，勾选“自动发送结果”，上位机会按识别结果发送一行中文文本：

```text
正常
划痕
污渍
```

每次发送末尾带换行，方便蓝牙模块或下位机按行读取。

说明：HC-05、BH207 等串口蓝牙模块在 Windows 底层仍会对应一个蓝牙串口，但界面上优先按设备名选择。模块和下位机之间的 UART 波特率需要在模块/下位机上匹配，上位机界面不再手动选择。若扫描到了设备但提示“未发现蓝牙串口”，请先在 Windows 蓝牙设置里完成配对，再回到上位机重新扫描。

## 3. OpenMV 端脚本

眼镜片采集脚本：

```text
openmv/n6_usb_image_capture.py
```

眼镜片模型检测脚本：

```text
openmv/n6_classifier_main.py
```

## 4. 训练和部署

在上位机“3 训练和部署”页可以启动本地训练、生成 YOLO 标注、训练 ONNX，并复制模型到 OpenMV N6。

默认资源：

```text
dataset/
models/lens_defect_classifier_int8.tflite
models/lens_defect_labels.txt
```

## 5. 打包 exe

```bat
windows_host\build_windows_exe.bat
```

打包后仍然只保留一个 exe：

```text
release/LensDefectHost.exe
```
