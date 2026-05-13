import csv
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None

try:
    from PIL import Image, ImageStat, ImageTk
except ImportError:
    Image = None
    ImageStat = None
    ImageTk = None


APP_TITLE = "OpenMV N6 眼镜片缺陷识别上位机"
DEFAULT_BAUDRATE = "115200"
READ_TIMEOUT_SECONDS = 0.2
CAPTURE_TIMEOUT_SECONDS = 12

APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent if APP_DIR.name in ("windows_host", "dist") else APP_DIR
HISTORY_DIR = APP_DIR / "history"
HISTORY_JSONL = HISTORY_DIR / "detection_history.jsonl"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
METADATA_FILENAME = "metadata.csv"
SPLIT_TARGET_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
MIN_RECOMMENDED_IMAGES_PER_CLASS = 100

DEFECT_TYPE_ORDER = [
    "normal",
    "scratch",
    "dust",
    "stain",
    "coating_damage",
    "crack",
    "edge_damage",
    "unknown",
]

SUMMARY_TYPES = [
    "scratch",
    "dust",
    "stain",
    "coating_damage",
    "crack",
    "edge_damage",
    "unknown",
]

DEFECT_TYPE_NAME = {
    "normal": "正常",
    "scratch": "划痕",
    "dust": "灰尘颗粒",
    "stain": "污点/油污",
    "coating_damage": "镀膜损伤",
    "crack": "裂纹",
    "edge_damage": "边缘损伤",
    "unknown": "未知缺陷",
}

LEVEL_NAME = {
    "normal": "正常",
    "light": "轻微",
    "medium": "中等",
    "serious": "严重",
}


@dataclass
class UiMessage:
    kind: str
    payload: object


def defect_type_name(value):
    return DEFECT_TYPE_NAME.get(value, value)


def level_name(value):
    return LEVEL_NAME.get(value, value)


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_filename_time():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def list_image_files(folder):
    if not folder.exists():
        return []
    return [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def count_image_files(folder):
    return len(list_image_files(folder))


def metadata_path(dataset_dir):
    return dataset_dir / METADATA_FILENAME


def find_training_script():
    candidates = [
        PROJECT_ROOT / "training" / "train_lens_classifier.py",
        APP_DIR / "training" / "train_lens_classifier.py",
        APP_DIR.parent / "training" / "train_lens_classifier.py",
        Path.cwd() / "training" / "train_lens_classifier.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class SerialLineReader:
    def __init__(self, ui_queue):
        self.ui_queue = ui_queue
        self.serial_port = None
        self.thread = None
        self.running = False

    def start(self, port, baudrate):
        if serial is None:
            raise RuntimeError("缺少 pyserial，请先运行 run_windows_host.bat 安装依赖。")
        if self.running:
            return

        self.serial_port = serial.Serial(
            port=port,
            baudrate=int(baudrate),
            timeout=READ_TIMEOUT_SECONDS,
        )
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception:
                pass
            self.serial_port = None

    def _read_loop(self):
        line_buffer = ""
        self.ui_queue.put(UiMessage("status", "串口已打开，正在接收检测 JSON"))

        while self.running:
            try:
                if self.serial_port is None:
                    break
                raw = self.serial_port.readline()
                if not raw:
                    continue
                line_buffer += raw.decode("utf-8", errors="ignore")

                while "\n" in line_buffer:
                    line, line_buffer = line_buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        self.ui_queue.put(UiMessage("line", line))
            except Exception as exc:
                self.ui_queue.put(UiMessage("error", "串口读取失败：%s" % exc))
                break

        self.running = False
        self.ui_queue.put(UiMessage("status", "检测接收已停止"))


class UsbImageCaptureClient:
    @staticmethod
    def capture(port, baudrate):
        if serial is None:
            raise RuntimeError("缺少 pyserial，请先运行 run_windows_host.bat 安装依赖。")

        with serial.Serial(port=port, baudrate=int(baudrate), timeout=0.25) as ser:
            time.sleep(0.25)
            ser.reset_input_buffer()
            ser.write(b"CAPTURE\n")
            ser.flush()

            size = None
            width = 0
            height = 0
            deadline = time.time() + CAPTURE_TIMEOUT_SECONDS

            while time.time() < deadline:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if line.startswith("ERR "):
                    raise RuntimeError(line[4:])
                if line.startswith("IMG_BEGIN "):
                    parts = line.split()
                    if len(parts) < 2:
                        raise RuntimeError("收到的图片头格式错误：%s" % line)
                    size = int(parts[1])
                    if len(parts) >= 4:
                        width = int(parts[2])
                        height = int(parts[3])
                    break

            if size is None:
                raise TimeoutError("等待 OpenMV 图片数据超时，请确认 N6 正在运行 n6_usb_image_capture.py")

            data = bytearray()
            while len(data) < size and time.time() < deadline:
                chunk = ser.read(size - len(data))
                if chunk:
                    data.extend(chunk)

            if len(data) != size:
                raise TimeoutError("图片数据不完整：收到 %d / %d 字节" % (len(data), size))

            end_line = ser.readline().decode("utf-8", errors="ignore").strip()
            if end_line and end_line != "IMG_END":
                # 旧固件或串口缓冲偶尔会吞掉结束行，图片本身完整时不强制失败。
                pass

            return bytes(data), width, height


class LensDefectHostApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1180x780")
        self.root.minsize(1020, 680)

        self.ui_queue = queue.Queue()
        self.reader = SerialLineReader(self.ui_queue)
        self.history_records = []
        self.auto_capture_running = False

        self.port_var = tk.StringVar()
        self.baud_var = tk.StringVar(value=DEFAULT_BAUDRATE)
        self.status_var = tk.StringVar(value="状态：未连接")

        self.has_defect_var = tk.StringVar(value="是否检测到缺陷：暂无数据")
        self.count_var = tk.StringVar(value="缺陷总数：0")
        self.level_var = tk.StringVar(value="整体严重程度：暂无数据")

        self.dataset_dir_var = tk.StringVar(value=str(DEFAULT_DATASET_DIR))
        self.dataset_class_var = tk.StringVar(value="normal")
        self.dataset_split_var = tk.StringVar(value="train")
        self.auto_split_var = tk.BooleanVar(value=True)
        self.capture_interval_var = tk.StringVar(value="1.0")
        self.capture_count_var = tk.StringVar(value="当前类别图片数：0")
        self.dataset_health_var = tk.StringVar(value="数据集体检：暂无统计")
        self.capture_preview_var = tk.StringVar(value="采集预览：暂无图片")
        self.capture_quality_var = tk.StringVar(value="质量提示：暂无图片")

        self.model_file_var = tk.StringVar(value=str(DEFAULT_MODELS_DIR / "lens_defect_classifier_int8.tflite"))
        self.labels_file_var = tk.StringVar(value=str(DEFAULT_MODELS_DIR / "lens_defect_labels.txt"))
        self.openmv_folder_var = tk.StringVar(value="")
        self.capture_preview_photo = None

        self._create_widgets()
        self.refresh_ports()
        self.load_history()
        self.refresh_dataset_counts()
        self.root.after(80, self._poll_ui_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _create_widgets(self):
        self._configure_style()

        root_frame = ttk.Frame(self.root, padding=12)
        root_frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(root_frame, text=APP_TITLE, style="Title.TLabel")
        title.pack(anchor="w")

        serial_frame = ttk.LabelFrame(root_frame, text="USB/串口连接", padding=10)
        serial_frame.pack(fill=tk.X, pady=(10, 8))

        ttk.Label(serial_frame, text="COM 口").pack(side=tk.LEFT)
        self.port_combo = ttk.Combobox(serial_frame, textvariable=self.port_var, width=32, state="readonly")
        self.port_combo.pack(side=tk.LEFT, padx=(8, 12))

        ttk.Label(serial_frame, text="波特率").pack(side=tk.LEFT)
        self.baud_combo = ttk.Combobox(
            serial_frame,
            textvariable=self.baud_var,
            width=10,
            values=("9600", "19200", "38400", "57600", "115200", "921600"),
        )
        self.baud_combo.pack(side=tk.LEFT, padx=(8, 12))

        ttk.Button(serial_frame, text="刷新串口", command=self.refresh_ports).pack(side=tk.LEFT, padx=4)
        ttk.Label(serial_frame, text="提示：OpenMV N6 USB 连接电脑后会出现一个 COM 口").pack(side=tk.LEFT, padx=12)

        ttk.Label(root_frame, textvariable=self.status_var, style="Status.TLabel").pack(fill=tk.X, pady=(0, 8))

        self.notebook = ttk.Notebook(root_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.detect_tab = ttk.Frame(self.notebook, padding=8)
        self.dataset_tab = ttk.Frame(self.notebook, padding=8)
        self.train_tab = ttk.Frame(self.notebook, padding=8)

        self.notebook.add(self.detect_tab, text="1 检测显示")
        self.notebook.add(self.dataset_tab, text="2 采集训练数据")
        self.notebook.add(self.train_tab, text="3 训练和部署")

        self._create_detect_tab()
        self._create_dataset_tab()
        self._create_train_tab()

    def _configure_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Status.TLabel", font=("Microsoft YaHei UI", 10), foreground="#1E6B5C")
        style.configure("Result.TLabel", font=("Microsoft YaHei UI", 11))
        style.configure("Treeview", rowheight=26, font=("Microsoft YaHei UI", 10))
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 10, "bold"))

    def _create_detect_tab(self):
        button_frame = ttk.Frame(self.detect_tab)
        button_frame.pack(fill=tk.X)
        ttk.Button(button_frame, text="开始接收检测 JSON", command=self.start_receive).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_frame, text="停止接收", command=self.stop_receive).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_frame, text="清空记录", command=self.clear_history).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_frame, text="导出 CSV", command=self.export_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_frame, text="模拟一条", command=self.mock_one_result).pack(side=tk.LEFT, padx=4)

        main_pane = ttk.PanedWindow(self.detect_tab, orient="horizontal")
        main_pane.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        left_frame = ttk.Frame(main_pane)
        right_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=3)
        main_pane.add(right_frame, weight=2)

        result_frame = ttk.LabelFrame(left_frame, text="当前检测结果", padding=10)
        result_frame.pack(fill=tk.X)
        ttk.Label(result_frame, textvariable=self.has_defect_var, style="Result.TLabel").pack(anchor="w")
        ttk.Label(result_frame, textvariable=self.count_var, style="Result.TLabel").pack(anchor="w", pady=3)
        ttk.Label(result_frame, textvariable=self.level_var, style="Result.TLabel").pack(anchor="w")

        summary_frame = ttk.LabelFrame(left_frame, text="各类缺陷数量", padding=8)
        summary_frame.pack(fill=tk.X, pady=(8, 8))
        self.summary_tree = ttk.Treeview(summary_frame, columns=("type", "count"), show="headings", height=7)
        self.summary_tree.heading("type", text="缺陷类型")
        self.summary_tree.heading("count", text="数量")
        self.summary_tree.column("type", width=160, anchor="center")
        self.summary_tree.column("count", width=80, anchor="center")
        self.summary_tree.pack(fill=tk.X)

        defect_frame = ttk.LabelFrame(left_frame, text="缺陷详情列表", padding=8)
        defect_frame.pack(fill=tk.BOTH, expand=True)
        self.defect_tree = ttk.Treeview(
            defect_frame,
            columns=("type", "confidence", "level", "x", "y", "w", "h", "area", "length", "ratio"),
            show="headings",
            height=12,
        )
        headings = {
            "type": "类型",
            "confidence": "置信度",
            "level": "等级",
            "x": "X",
            "y": "Y",
            "w": "W",
            "h": "H",
            "area": "面积",
            "length": "长度",
            "ratio": "长宽比",
        }
        widths = {
            "type": 95,
            "confidence": 70,
            "level": 60,
            "x": 45,
            "y": 45,
            "w": 45,
            "h": 45,
            "area": 70,
            "length": 70,
            "ratio": 70,
        }
        for column, text in headings.items():
            self.defect_tree.heading(column, text=text)
            self.defect_tree.column(column, width=widths[column], anchor="center")
        self.defect_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        defect_scroll = ttk.Scrollbar(defect_frame, orient="vertical", command=self.defect_tree.yview)
        defect_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.defect_tree.configure(yscrollcommand=defect_scroll.set)

        raw_frame = ttk.LabelFrame(right_frame, text="最近一次 JSON 原始数据", padding=8)
        raw_frame.pack(fill=tk.BOTH, expand=True)
        self.raw_text = self._create_text(raw_frame, height=12)

        history_frame = ttk.LabelFrame(right_frame, text="历史检测记录", padding=8)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.history_tree = ttk.Treeview(
            history_frame,
            columns=("time", "status", "count", "level", "timestamp"),
            show="headings",
            height=13,
        )
        for column, text, width in (
            ("time", "接收时间", 150),
            ("status", "结果", 80),
            ("count", "数量", 50),
            ("level", "等级", 60),
            ("timestamp", "OpenMV 时间戳", 110),
        ):
            self.history_tree.heading(column, text=text)
            self.history_tree.column(column, width=width, anchor="center")
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scroll = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_tree.yview)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_tree.configure(yscrollcommand=history_scroll.set)

        self._reset_summary()

    def _create_dataset_tab(self):
        config_frame = ttk.LabelFrame(self.dataset_tab, text="数据集保存设置", padding=10)
        config_frame.pack(fill=tk.X)

        ttk.Label(config_frame, text="数据集目录").grid(row=0, column=0, sticky="w")
        ttk.Entry(config_frame, textvariable=self.dataset_dir_var, width=72).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(config_frame, text="选择目录", command=self.choose_dataset_dir).grid(row=0, column=2, padx=4)
        ttk.Button(config_frame, text="初始化分类文件夹", command=self.init_dataset_folders).grid(row=0, column=3, padx=4)

        ttk.Label(config_frame, text="类别").grid(row=1, column=0, sticky="w", pady=(8, 0))
        class_combo = ttk.Combobox(
            config_frame,
            textvariable=self.dataset_class_var,
            values=DEFECT_TYPE_ORDER,
            state="readonly",
            width=20,
        )
        class_combo.grid(row=1, column=1, sticky="w", padx=8, pady=(8, 0))
        class_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_dataset_counts())

        ttk.Label(config_frame, text="集合").grid(row=1, column=1, sticky="w", padx=(210, 0), pady=(8, 0))
        ttk.Combobox(
            config_frame,
            textvariable=self.dataset_split_var,
            values=("train", "val", "test"),
            state="readonly",
            width=10,
        ).grid(row=1, column=1, sticky="w", padx=(250, 0), pady=(8, 0))

        ttk.Label(config_frame, text="自动采集间隔秒").grid(row=1, column=2, sticky="e", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.capture_interval_var, width=8).grid(row=1, column=3, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            config_frame,
            text="按 70/20/10 自动分配 train/val/test",
            variable=self.auto_split_var,
        ).grid(row=2, column=1, sticky="w", padx=8, pady=(8, 0))
        config_frame.columnconfigure(1, weight=1)

        action_frame = ttk.Frame(self.dataset_tab)
        action_frame.pack(fill=tk.X, pady=8)
        ttk.Button(action_frame, text="测试采图连接", command=self.test_capture_connection).pack(side=tk.LEFT, padx=4)
        ttk.Button(action_frame, text="拍一张保存到电脑", command=self.capture_one_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(action_frame, text="开始自动采集", command=self.start_auto_capture).pack(side=tk.LEFT, padx=4)
        ttk.Button(action_frame, text="停止自动采集", command=self.stop_auto_capture).pack(side=tk.LEFT, padx=4)
        ttk.Button(action_frame, text="打开数据集目录", command=self.open_dataset_dir).pack(side=tk.LEFT, padx=4)
        ttk.Button(action_frame, text="刷新统计", command=self.refresh_dataset_counts).pack(side=tk.LEFT, padx=4)
        ttk.Label(action_frame, textvariable=self.capture_count_var, style="Status.TLabel").pack(side=tk.LEFT, padx=12)

        pane = ttk.PanedWindow(self.dataset_tab, orient="horizontal")
        pane.pack(fill=tk.BOTH, expand=True)

        count_frame = ttk.LabelFrame(pane, text="数据集数量统计", padding=8)
        right_panel = ttk.Frame(pane)
        pane.add(count_frame, weight=1)
        pane.add(right_panel, weight=2)

        self.dataset_tree = ttk.Treeview(count_frame, columns=("class", "train", "val", "test", "total"), show="headings")
        for column, text, width in (
            ("class", "类别", 120),
            ("train", "train", 70),
            ("val", "val", 70),
            ("test", "test", 70),
            ("total", "总数", 70),
        ):
            self.dataset_tree.heading(column, text=text)
            self.dataset_tree.column(column, width=width, anchor="center")
        self.dataset_tree.pack(fill=tk.BOTH, expand=True)
        ttk.Label(count_frame, textvariable=self.dataset_health_var, wraplength=360, style="Status.TLabel").pack(fill=tk.X, pady=(8, 0))

        preview_frame = ttk.LabelFrame(right_panel, text="最近采集预览和质量提示", padding=8)
        preview_frame.pack(fill=tk.X)
        self.capture_preview_label = tk.Label(
            preview_frame,
            textvariable=self.capture_preview_var,
            compound="top",
            bg="#F8F8F8",
            relief="groove",
            width=52,
            height=13,
            anchor="center",
            justify="center",
        )
        self.capture_preview_label.pack(fill=tk.X)
        ttk.Label(preview_frame, textvariable=self.capture_quality_var, wraplength=620, style="Status.TLabel").pack(fill=tk.X, pady=(6, 0))

        log_frame = ttk.LabelFrame(right_panel, text="采集日志", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.capture_log_text = self._create_text(log_frame, height=14)
        self._set_text(self.capture_log_text, "操作步骤：\n1. 把 openmv/n6_usb_image_capture.py 保存到 OpenMV N6 并运行。\n2. 关闭 OpenMV IDE 串口占用，或确保上位机能打开 N6 的 COM 口。\n3. 选择类别，建议保持“自动分配集合”开启后开始采集。")

    def _create_train_tab(self):
        workflow = (
            "推荐流程：OpenMV N6 拍图 -> USB 连接电脑 -> 电脑保存训练图片 -> 电脑训练模型 "
            "-> 选择 OpenMV 盘符/文件夹 -> 拷贝 .tflite 和 labels.txt -> N6 运行模型脚本。"
        )
        ttk.Label(self.train_tab, text=workflow, wraplength=1000, style="Result.TLabel").pack(anchor="w", pady=(0, 8))

        train_frame = ttk.LabelFrame(self.train_tab, text="电脑训练模型", padding=10)
        train_frame.pack(fill=tk.X)
        ttk.Button(train_frame, text="打开数据集目录", command=self.open_dataset_dir).pack(side=tk.LEFT, padx=4)
        ttk.Button(train_frame, text="打开模型输出目录", command=self.open_models_dir).pack(side=tk.LEFT, padx=4)
        ttk.Button(train_frame, text="启动本地训练脚本", command=self.run_training_script).pack(side=tk.LEFT, padx=4)
        ttk.Button(train_frame, text="打开训练脚本位置", command=self.open_training_script_folder).pack(side=tk.LEFT, padx=4)
        ttk.Button(train_frame, text="刷新训练结果", command=self.load_training_summary).pack(side=tk.LEFT, padx=4)

        deploy_frame = ttk.LabelFrame(self.train_tab, text="把模型文件传回 OpenMV N6", padding=10)
        deploy_frame.pack(fill=tk.X, pady=(10, 8))
        deploy_frame.columnconfigure(1, weight=1)

        ttk.Label(deploy_frame, text="模型 .tflite").grid(row=0, column=0, sticky="w")
        ttk.Entry(deploy_frame, textvariable=self.model_file_var).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(deploy_frame, text="选择模型", command=self.choose_model_file).grid(row=0, column=2, padx=4)

        ttk.Label(deploy_frame, text="标签 .txt").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(deploy_frame, textvariable=self.labels_file_var).grid(row=1, column=1, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(deploy_frame, text="选择标签", command=self.choose_labels_file).grid(row=1, column=2, padx=4, pady=(8, 0))

        ttk.Label(deploy_frame, text="OpenMV 盘符/目录").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(deploy_frame, textvariable=self.openmv_folder_var).grid(row=2, column=1, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(deploy_frame, text="选择目录", command=self.choose_openmv_folder).grid(row=2, column=2, padx=4, pady=(8, 0))

        ttk.Button(deploy_frame, text="复制模型到 OpenMV N6", command=self.copy_model_to_openmv).grid(row=3, column=1, sticky="w", padx=8, pady=(12, 0))

        result_frame = ttk.LabelFrame(self.train_tab, text="最近训练结果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self.training_result_text = self._create_text(result_frame, height=9)

        script_frame = ttk.LabelFrame(self.train_tab, text="N6 端脚本提示", padding=10)
        script_frame.pack(fill=tk.X)
        text = (
            "采集数据时：运行 openmv/n6_usb_image_capture.py。\n"
            "部署模型后：运行 openmv/n6_classifier_main.py，它会加载 /lens_defect_classifier_int8.tflite 和 /lens_defect_labels.txt。\n"
            "如果模型较大，建议之后加一张 32GB microSD，把模型放 /sd/ 目录。"
        )
        ttk.Label(script_frame, text=text, wraplength=1000).pack(anchor="w")
        self.load_training_summary()

    def _create_text(self, parent, height):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)
        widget = tk.Text(frame, height=height, wrap="word", font=("Consolas", 10))
        widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=widget.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        widget.configure(yscrollcommand=scroll.set)
        widget.insert("1.0", "暂无数据")
        widget.configure(state="disabled")
        return widget

    def refresh_ports(self):
        if list_ports is None:
            self.status_var.set("状态：缺少 pyserial，运行 run_windows_host.bat 会自动安装依赖")
            return

        ports = []
        for port in list_ports.comports():
            ports.append("%s - %s" % (port.device, port.description))

        self.port_combo["values"] = ports
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])
            self.status_var.set("状态：找到 %d 个串口，请选择 OpenMV N6 对应 COM 口" % len(ports))
        elif not ports:
            self.port_var.set("")
            self.status_var.set("状态：未找到串口。请用 USB 连接 OpenMV N6")

    def selected_port(self):
        selected = self.port_var.get().strip()
        if not selected:
            raise RuntimeError("请先选择 COM 口。")
        return selected.split(" - ", 1)[0].strip()

    def selected_baudrate(self):
        baudrate = self.baud_var.get().strip()
        int(baudrate)
        return baudrate

    def start_receive(self):
        if self.reader.running:
            messagebox.showinfo("提示", "当前已经在接收检测 JSON。")
            return
        try:
            self.reader.start(self.selected_port(), self.selected_baudrate())
            self.status_var.set("状态：正在接收 OpenMV 检测 JSON")
        except Exception as exc:
            messagebox.showerror("连接失败", str(exc))
            self.status_var.set("状态：连接失败")

    def stop_receive(self):
        self.reader.stop()
        self.status_var.set("状态：已停止接收")

    def choose_dataset_dir(self):
        path = filedialog.askdirectory(title="选择数据集目录", initialdir=self.dataset_dir_var.get())
        if path:
            self.dataset_dir_var.set(path)
            self.refresh_dataset_counts()

    def init_dataset_folders(self):
        dataset_dir = Path(self.dataset_dir_var.get())
        for split in ("train", "val", "test"):
            for class_name in DEFECT_TYPE_ORDER:
                (dataset_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        self.refresh_dataset_counts()
        self.log_capture("已初始化数据集目录：%s" % dataset_dir)

    def test_capture_connection(self):
        try:
            self.capture_one_image(test_only=True)
            messagebox.showinfo("成功", "已收到 OpenMV N6 发来的图片数据，连接正常。")
        except Exception as exc:
            messagebox.showerror("测试失败", str(exc))

    def capture_one_image(self, test_only=False):
        try:
            settings = self.capture_settings()
            payload = self.capture_and_store_image(settings, test_only=test_only)
            self.apply_capture_result(payload)
        except Exception as exc:
            if test_only:
                raise
            self.status_var.set("状态：采集失败")
            self.log_capture("采集失败：%s" % exc)
            messagebox.showerror("采集失败", str(exc))

    def start_auto_capture(self):
        if self.auto_capture_running:
            messagebox.showinfo("提示", "自动采集已经在运行。")
            return
        try:
            interval = float(self.capture_interval_var.get())
            if interval <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "自动采集间隔必须是大于 0 的数字。")
            return

        try:
            settings = self.capture_settings()
        except Exception as exc:
            messagebox.showerror("采集失败", str(exc))
            return

        self.auto_capture_running = True
        self.status_var.set("状态：开始自动采集训练图片")
        threading.Thread(target=self._auto_capture_loop, args=(interval, settings), daemon=True).start()

    def stop_auto_capture(self):
        self.auto_capture_running = False
        self.status_var.set("状态：已请求停止自动采集")

    def _auto_capture_loop(self, interval, settings):
        while self.auto_capture_running:
            try:
                payload = self.capture_and_store_image(settings, test_only=False)
                self.ui_queue.put(UiMessage("capture_saved", payload))
            except Exception as exc:
                self.ui_queue.put(UiMessage("capture_error", str(exc)))
                self.auto_capture_running = False
                break
            time.sleep(interval)
        self.ui_queue.put(UiMessage("status", "自动采集已停止"))

    def capture_settings(self):
        if self.reader.running:
            raise RuntimeError("检测接收正在占用串口，请先停止接收。")
        return {
            "dataset_dir": Path(self.dataset_dir_var.get()),
            "class_name": self.dataset_class_var.get(),
            "split": self.dataset_split_var.get(),
            "auto_split": bool(self.auto_split_var.get()),
            "port": self.selected_port(),
            "baudrate": self.selected_baudrate(),
        }

    def capture_and_store_image(self, settings, test_only=False):
        dataset_dir = settings["dataset_dir"]
        class_name = settings["class_name"]
        split = settings["split"]
        port = settings["port"]
        baudrate = settings["baudrate"]

        image_bytes, width, height = UsbImageCaptureClient.capture(port, baudrate)
        quality = self.inspect_capture_image(image_bytes, width, height)

        save_path = None
        if not test_only:
            if settings["auto_split"]:
                split = self.choose_balanced_split(dataset_dir, class_name)
            save_dir = dataset_dir / split / class_name
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = "%s_%s_%s.jpg" % (class_name, split, safe_filename_time())
            save_path = save_dir / filename
            with open(save_path, "wb") as file:
                file.write(image_bytes)
            self.append_capture_metadata(
                dataset_dir=dataset_dir,
                save_path=save_path,
                split=split,
                class_name=class_name,
                port=port,
                baudrate=baudrate,
                width=quality["width"],
                height=quality["height"],
                byte_count=len(image_bytes),
                quality=quality,
            )

        return {
            "test_only": test_only,
            "save_path": str(save_path) if save_path else "",
            "split": split,
            "class_name": class_name,
            "byte_count": len(image_bytes),
            "width": quality["width"],
            "height": quality["height"],
            "image_bytes": image_bytes,
            "quality": quality,
        }

    def choose_balanced_split(self, dataset_dir, class_name):
        counts = {}
        for split in ("train", "val", "test"):
            counts[split] = count_image_files(dataset_dir / split / class_name)
        total_after = sum(counts.values()) + 1

        best_split = "train"
        best_score = None
        for candidate in ("train", "val", "test"):
            candidate_counts = dict(counts)
            candidate_counts[candidate] += 1
            score = 0.0
            for split, ratio in SPLIT_TARGET_RATIOS.items():
                actual = candidate_counts[split] / total_after
                score += abs(actual - ratio)
            if best_score is None or score < best_score:
                best_split = candidate
                best_score = score
        return best_split

    def inspect_capture_image(self, image_bytes, width, height):
        quality = {
            "width": width,
            "height": height,
            "brightness": "",
            "contrast": "",
            "status": "未检查",
            "warnings": [],
        }

        if Image is None or ImageStat is None:
            quality["warnings"].append("未安装 Pillow，只保存图片，不显示 JPEG 预览和亮度检查")
            return quality

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                quality["width"], quality["height"] = image.size
                gray = image.convert("L")
                stat = ImageStat.Stat(gray)
                brightness = float(stat.mean[0])
                contrast = float(stat.stddev[0])
                quality["brightness"] = "%.1f" % brightness
                quality["contrast"] = "%.1f" % contrast

                if brightness < 35:
                    quality["warnings"].append("图片偏暗，建议增加补光或曝光")
                elif brightness > 220:
                    quality["warnings"].append("图片偏亮，缺陷细节可能被过曝淹没")
                if contrast < 12:
                    quality["warnings"].append("对比度偏低，透明镜片缺陷可能不明显")
                if min(image.size) < 96:
                    quality["warnings"].append("图片尺寸偏小，建议保持至少 128x128 以上")
        except Exception as exc:
            quality["warnings"].append("图片质量检查失败：%s" % exc)

        quality["status"] = "通过" if not quality["warnings"] else "需复核"
        return quality

    def append_capture_metadata(self, dataset_dir, save_path, split, class_name, port, baudrate, width, height, byte_count, quality):
        dataset_dir.mkdir(parents=True, exist_ok=True)
        path = metadata_path(dataset_dir)
        is_new_file = not path.exists()
        try:
            relative_path = str(save_path.relative_to(dataset_dir))
        except ValueError:
            relative_path = str(save_path)

        with open(path, "a", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            if is_new_file:
                writer.writerow([
                    "capture_time",
                    "relative_path",
                    "split",
                    "label",
                    "width",
                    "height",
                    "bytes",
                    "brightness",
                    "contrast",
                    "quality_status",
                    "quality_warnings",
                    "port",
                    "baudrate",
                    "source",
                ])
            writer.writerow([
                now_text(),
                relative_path,
                split,
                class_name,
                width,
                height,
                byte_count,
                quality.get("brightness", ""),
                quality.get("contrast", ""),
                quality.get("status", ""),
                "；".join(quality.get("warnings", [])),
                port,
                baudrate,
                "OpenMV N6 USB",
            ])

    def apply_capture_result(self, payload):
        quality = payload["quality"]
        quality_text = self.format_quality_text(quality)
        if payload["test_only"]:
            self.log_capture(
                "测试采图成功：%d 字节，尺寸 %dx%d，%s"
                % (payload["byte_count"], payload["width"], payload["height"], quality_text)
            )
        else:
            self.log_capture(
                "保存图片：%s，类别 %s，集合 %s，尺寸 %dx%d，%s"
                % (
                    payload["save_path"],
                    defect_type_name(payload["class_name"]),
                    payload["split"],
                    payload["width"],
                    payload["height"],
                    quality_text,
                )
            )
            self.refresh_dataset_counts()
        self.update_capture_preview(payload)

    def format_quality_text(self, quality):
        warnings = quality.get("warnings") or []
        base = "质量：%s" % quality.get("status", "未检查")
        if quality.get("brightness"):
            base += "，亮度 %s" % quality["brightness"]
        if quality.get("contrast"):
            base += "，对比度 %s" % quality["contrast"]
        if warnings:
            base += "；" + "；".join(warnings)
        return base

    def update_capture_preview(self, payload):
        quality_text = self.format_quality_text(payload["quality"])
        location = "测试采图，未保存"
        if payload["save_path"]:
            location = "%s / %s" % (payload["split"], defect_type_name(payload["class_name"]))
        self.capture_preview_var.set(
            "最近图片：%s\n尺寸：%dx%d，大小：%d 字节"
            % (location, payload["width"], payload["height"], payload["byte_count"])
        )
        self.capture_quality_var.set(quality_text)

        if Image is None or ImageTk is None:
            self.capture_preview_label.configure(image="")
            return

        try:
            with Image.open(BytesIO(payload["image_bytes"])) as image:
                image.thumbnail((520, 220))
                self.capture_preview_photo = ImageTk.PhotoImage(image.copy())
            self.capture_preview_label.configure(image=self.capture_preview_photo, compound="top")
        except Exception as exc:
            self.capture_preview_photo = None
            self.capture_preview_label.configure(image="")
            self.capture_quality_var.set(quality_text + "；预览失败：%s" % exc)

    def refresh_dataset_counts(self):
        if not hasattr(self, "dataset_tree"):
            return
        dataset_dir = Path(self.dataset_dir_var.get())
        self.dataset_tree.delete(*self.dataset_tree.get_children())

        selected_class = self.dataset_class_var.get()
        selected_total = 0
        totals = {}
        split_missing_classes = []
        for class_name in DEFECT_TYPE_ORDER:
            counts = []
            total = 0
            for split in ("train", "val", "test"):
                folder = dataset_dir / split / class_name
                count = count_image_files(folder)
                counts.append(count)
                total += count
            if class_name == selected_class:
                selected_total = total
            totals[class_name] = total
            if total > 0 and (counts[1] == 0 or counts[2] == 0):
                split_missing_classes.append(defect_type_name(class_name))
            self.dataset_tree.insert("", tk.END, values=(defect_type_name(class_name), counts[0], counts[1], counts[2], total))

        self.capture_count_var.set("当前类别图片数：%d" % selected_total)
        self.dataset_health_var.set(self.build_dataset_health_text(totals, split_missing_classes))

    def build_dataset_health_text(self, totals, split_missing_classes):
        image_total = sum(totals.values())
        active_classes = [class_name for class_name, total in totals.items() if total > 0]
        low_classes = [
            defect_type_name(class_name)
            for class_name, total in totals.items()
            if 0 < total < MIN_RECOMMENDED_IMAGES_PER_CLASS
        ]

        messages = ["数据集体检：共 %d 张，有数据类别 %d/%d" % (image_total, len(active_classes), len(DEFECT_TYPE_ORDER))]
        if low_classes:
            messages.append("不足 %d 张：%s" % (MIN_RECOMMENDED_IMAGES_PER_CLASS, "、".join(low_classes[:5])))
        if split_missing_classes:
            messages.append("缺少 val/test：%s" % "、".join(split_missing_classes[:5]))

        non_zero_totals = [total for total in totals.values() if total > 0]
        if len(non_zero_totals) >= 2 and min(non_zero_totals) > 0 and max(non_zero_totals) / min(non_zero_totals) > 3:
            messages.append("类别数量差异较大，训练可能偏向样本多的类别")
        if image_total == 0:
            messages.append("建议先每类至少采 100 张，稳定后再扩到 300 张以上")

        metadata = metadata_path(Path(self.dataset_dir_var.get()))
        if metadata.exists():
            messages.append("已记录 %s" % metadata.name)
        else:
            messages.append("下一次采集会自动生成 metadata.csv")

        return "；".join(messages)

    def open_dataset_dir(self):
        path = Path(self.dataset_dir_var.get())
        path.mkdir(parents=True, exist_ok=True)
        self.open_path(path)

    def open_models_dir(self):
        DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.open_path(DEFAULT_MODELS_DIR)

    def load_training_summary(self):
        if not hasattr(self, "training_result_text"):
            return

        summary_path = DEFAULT_MODELS_DIR / "training_summary.json"
        if not summary_path.exists():
            self._set_text(
                self.training_result_text,
                "暂无训练结果。训练结束后这里会读取 models/training_summary.json、training_history.csv 和混淆矩阵文件。",
            )
            return

        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._set_text(self.training_result_text, "训练结果读取失败：%s" % exc)
            return

        lines = []
        lines.append("数据集：%s" % summary.get("dataset", ""))
        lines.append("输入尺寸：%s x %s" % (summary.get("image_size", ""), summary.get("image_size", "")))
        lines.append("类别顺序：%s" % "、".join(summary.get("labels", [])))
        metrics = summary.get("metrics", {})
        for split in ("train", "val", "test"):
            values = metrics.get(split)
            if not values:
                continue
            accuracy = values.get("accuracy")
            loss = values.get("loss")
            lines.append("%s：accuracy=%s，loss=%s" % (split, self.format_metric(accuracy), self.format_metric(loss)))

        artifacts = summary.get("artifacts", {})
        if artifacts.get("int8_tflite"):
            lines.append("INT8 模型：%s" % artifacts["int8_tflite"])
        if artifacts.get("confusion_matrix"):
            lines.append("混淆矩阵：%s" % artifacts["confusion_matrix"])
        lines.append("更新时间：%s" % now_text())
        self._set_text(self.training_result_text, "\n".join(lines))

    def format_metric(self, value):
        if value is None:
            return "暂无"
        try:
            return "%.4f" % float(value)
        except (TypeError, ValueError):
            return str(value)

    def run_training_script(self):
        script = find_training_script()
        if script is None:
            messagebox.showerror("找不到训练脚本", "未找到 training/train_lens_classifier.py")
            return

        dataset_dir = Path(self.dataset_dir_var.get())
        DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        command = (
            'python "%s" --dataset "%s" --output "%s" --epochs 20 --image-size 128'
            % (script, dataset_dir, DEFAULT_MODELS_DIR)
        )
        subprocess.Popen(["cmd", "/k", command], cwd=str(PROJECT_ROOT))
        self.status_var.set("状态：已打开训练命令窗口")

    def open_training_script_folder(self):
        script = find_training_script()
        if script is None:
            messagebox.showerror("找不到训练脚本", "未找到 training/train_lens_classifier.py")
            return
        self.open_path(script.parent)

    def choose_model_file(self):
        path = filedialog.askopenfilename(
            title="选择 TFLite 模型",
            filetypes=(("TFLite 模型", "*.tflite"), ("所有文件", "*.*")),
        )
        if path:
            self.model_file_var.set(path)

    def choose_labels_file(self):
        path = filedialog.askopenfilename(
            title="选择标签文件",
            filetypes=(("文本文件", "*.txt"), ("所有文件", "*.*")),
        )
        if path:
            self.labels_file_var.set(path)

    def choose_openmv_folder(self):
        path = filedialog.askdirectory(title="选择 OpenMV N6 USB 盘符或文件夹")
        if path:
            self.openmv_folder_var.set(path)

    def copy_model_to_openmv(self):
        model_path = Path(self.model_file_var.get())
        labels_path = Path(self.labels_file_var.get())
        target_dir = Path(self.openmv_folder_var.get())

        if not model_path.exists():
            messagebox.showerror("错误", "模型文件不存在：%s" % model_path)
            return
        if not labels_path.exists():
            messagebox.showerror("错误", "标签文件不存在：%s" % labels_path)
            return
        if not target_dir.exists():
            messagebox.showerror("错误", "OpenMV 目标目录不存在：%s" % target_dir)
            return

        target_model = target_dir / "lens_defect_classifier_int8.tflite"
        target_labels = target_dir / "lens_defect_labels.txt"
        shutil.copy2(model_path, target_model)
        shutil.copy2(labels_path, target_labels)
        messagebox.showinfo("完成", "已复制：\n%s\n%s" % (target_model, target_labels))
        self.status_var.set("状态：模型文件已复制到 OpenMV N6")

    def open_path(self, path):
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))

    def log_capture(self, message):
        old = self.capture_log_text.get("1.0", tk.END).strip()
        new_text = "%s  %s" % (now_text(), message)
        if old and old != "暂无数据":
            new_text = old + "\n" + new_text
        self._set_text(self.capture_log_text, new_text)

    def _poll_ui_queue(self):
        while True:
            try:
                message = self.ui_queue.get_nowait()
            except queue.Empty:
                break

            if message.kind == "line":
                self.handle_json_line(message.payload)
            elif message.kind == "status":
                self.status_var.set("状态：" + str(message.payload))
            elif message.kind == "error":
                self.status_var.set("状态：" + str(message.payload))
                messagebox.showerror("串口错误", str(message.payload))
            elif message.kind == "capture_error":
                self.status_var.set("状态：采集失败")
                self.log_capture("采集失败：%s" % message.payload)
                messagebox.showerror("采集失败", str(message.payload))
            elif message.kind == "capture_saved":
                self.apply_capture_result(message.payload)

        self.root.after(80, self._poll_ui_queue)

    def handle_json_line(self, raw_line):
        self._set_raw_text(raw_line)
        try:
            result = json.loads(raw_line)
            if not isinstance(result, dict):
                raise ValueError("JSON 顶层必须是对象")
        except Exception as exc:
            self.status_var.set("状态：JSON 解析失败：%s" % exc)
            return

        self.update_result_ui(result)
        self.add_history(result, raw_line)
        self.status_var.set("状态：%s 收到一条有效 JSON" % now_text())

    def update_result_ui(self, result):
        has_defect = bool(result.get("has_defect", False))
        defect_count = int(result.get("defect_count", 0))
        overall_level = result.get("overall_level", "normal")

        self.has_defect_var.set("是否检测到缺陷：%s" % ("是" if has_defect else "否"))
        self.count_var.set("缺陷总数：%d" % defect_count)
        self.level_var.set("整体严重程度：%s" % level_name(overall_level))

        self.summary_tree.delete(*self.summary_tree.get_children())
        summary = result.get("summary") or {}
        for defect_type in SUMMARY_TYPES:
            self.summary_tree.insert("", tk.END, values=(defect_type_name(defect_type), summary.get(defect_type, 0)))

        self.defect_tree.delete(*self.defect_tree.get_children())
        defects = result.get("defects") or []
        for defect in defects:
            self.defect_tree.insert(
                "",
                tk.END,
                values=(
                    defect_type_name(defect.get("type", "unknown")),
                    "%.2f" % float(defect.get("confidence", 0)),
                    level_name(defect.get("level", "normal")),
                    defect.get("x", 0),
                    defect.get("y", 0),
                    defect.get("w", 0),
                    defect.get("h", 0),
                    defect.get("area", 0),
                    defect.get("length", 0),
                    "%.2f" % float(defect.get("aspect_ratio", 0)),
                ),
            )

    def add_history(self, result, raw_line):
        record = {
            "receive_time": now_text(),
            "result": result,
            "raw_json": raw_line,
        }
        self.history_records.insert(0, record)
        self.history_records = self.history_records[:300]
        self._append_history_file(record)
        self._insert_history_row(record)

    def _insert_history_row(self, record):
        result = record["result"]
        status = "有缺陷" if result.get("has_defect") else "正常"
        values = (
            record["receive_time"],
            status,
            result.get("defect_count", 0),
            level_name(result.get("overall_level", "normal")),
            result.get("timestamp", ""),
        )
        self.history_tree.insert("", 0, values=values)

    def load_history(self):
        if not HISTORY_JSONL.exists():
            return

        try:
            with open(HISTORY_JSONL, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self.history_records.insert(0, record)
        except Exception:
            self.history_records.clear()

        self.history_records = self.history_records[:300]
        for record in reversed(self.history_records):
            self._insert_history_row(record)

    def _append_history_file(self, record):
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_JSONL, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def clear_history(self):
        if not messagebox.askyesno("确认", "确定清空界面和历史记录吗？"):
            return
        self.history_records.clear()
        self.history_tree.delete(*self.history_tree.get_children())
        self.defect_tree.delete(*self.defect_tree.get_children())
        self._reset_summary()
        self.has_defect_var.set("是否检测到缺陷：暂无数据")
        self.count_var.set("缺陷总数：0")
        self.level_var.set("整体严重程度：暂无数据")
        self._set_raw_text("暂无数据")
        if HISTORY_JSONL.exists():
            HISTORY_JSONL.unlink()

    def export_csv(self):
        if not self.history_records:
            messagebox.showinfo("提示", "暂无历史记录可导出。")
            return

        default_name = "lens_defect_history_%s.csv" % datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            title="导出历史记录",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=(("CSV 文件", "*.csv"), ("所有文件", "*.*")),
        )
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["接收时间", "是否有缺陷", "缺陷数量", "整体等级", "OpenMV时间戳", "原始JSON"])
            for record in self.history_records:
                writer.writerow([
                    record["receive_time"],
                    "是" if record["result"].get("has_defect") else "否",
                    record["result"].get("defect_count", 0),
                    level_name(record["result"].get("overall_level", "normal")),
                    record["result"].get("timestamp", ""),
                    record["raw_json"],
                ])
        messagebox.showinfo("完成", "已导出：%s" % path)

    def mock_one_result(self):
        sample = {
            "has_defect": True,
            "defect_count": 3,
            "summary": {
                "scratch": 1,
                "dust": 1,
                "stain": 1,
                "coating_damage": 0,
                "crack": 0,
                "edge_damage": 0,
                "unknown": 0,
            },
            "overall_level": "medium",
            "defects": [
                {"type": "scratch", "confidence": 0.86, "x": 120, "y": 80, "w": 70, "h": 5, "area": 350, "length": 70, "aspect_ratio": 14.0, "level": "medium"},
                {"type": "dust", "confidence": 0.75, "x": 210, "y": 130, "w": 6, "h": 5, "area": 30, "length": 6, "aspect_ratio": 1.2, "level": "light"},
                {"type": "stain", "confidence": 0.68, "x": 88, "y": 112, "w": 22, "h": 18, "area": 180, "length": 22, "aspect_ratio": 1.22, "level": "light"},
            ],
            "timestamp": int(time.time() * 1000) % 10000000,
        }
        self.handle_json_line(json.dumps(sample, ensure_ascii=False))

    def _reset_summary(self):
        self.summary_tree.delete(*self.summary_tree.get_children())
        for defect_type in SUMMARY_TYPES:
            self.summary_tree.insert("", tk.END, values=(defect_type_name(defect_type), 0))

    def _set_raw_text(self, value):
        pretty_value = value
        try:
            pretty_value = json.dumps(json.loads(value), ensure_ascii=False, indent=2)
        except Exception:
            pass
        self._set_text(self.raw_text, pretty_value)

    def _set_text(self, widget, value):
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def on_close(self):
        self.auto_capture_running = False
        self.reader.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    LensDefectHostApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
