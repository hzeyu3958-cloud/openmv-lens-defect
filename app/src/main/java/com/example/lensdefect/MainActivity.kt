package com.example.lensdefect

import android.Manifest
import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothSocket
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ListView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import org.json.JSONArray
import java.io.IOException
import java.io.InputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    private lateinit var statusTextView: TextView
    private lateinit var deviceListView: ListView
    private lateinit var connectButton: Button
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var clearButton: Button
    private lateinit var defectStatusTextView: TextView
    private lateinit var defectCountTextView: TextView
    private lateinit var overallLevelTextView: TextView
    private lateinit var summaryTextView: TextView
    private lateinit var rawJsonTextView: TextView
    private lateinit var historyTextView: TextView
    private lateinit var defectRecyclerView: RecyclerView

    private val defectAdapter = DefectAdapter()
    private val pairedDevices = mutableListOf<BluetoothDevice>()
    private val deviceNames = mutableListOf<String>()
    private val historyRecords = mutableListOf<String>()

    private var bluetoothAdapter: BluetoothAdapter? = null
    private var selectedDevice: BluetoothDevice? = null
    private var socket: BluetoothSocket? = null
    private var inputStream: InputStream? = null

    @Volatile
    private var receiving = false

    companion object {
        private const val REQUEST_PERMISSIONS = 1001
        private const val REQUEST_ENABLE_BLUETOOTH = 1002
        private const val HISTORY_PREF = "detection_history"
        private const val HISTORY_KEY = "records"
        private const val MAX_HISTORY = 100
        private val SPP_UUID: UUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        bindViews()
        setupRecyclerView()
        setupBluetooth()
        setupButtons()
        loadHistory()

        if (hasRequiredPermissions()) {
            loadPairedDevices()
        } else {
            requestRequiredPermissions()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopReceiving()
        closeSocket()
    }

    private fun bindViews() {
        statusTextView = findViewById(R.id.statusTextView)
        deviceListView = findViewById(R.id.deviceListView)
        connectButton = findViewById(R.id.connectButton)
        startButton = findViewById(R.id.startButton)
        stopButton = findViewById(R.id.stopButton)
        clearButton = findViewById(R.id.clearButton)
        defectStatusTextView = findViewById(R.id.defectStatusTextView)
        defectCountTextView = findViewById(R.id.defectCountTextView)
        overallLevelTextView = findViewById(R.id.overallLevelTextView)
        summaryTextView = findViewById(R.id.summaryTextView)
        rawJsonTextView = findViewById(R.id.rawJsonTextView)
        historyTextView = findViewById(R.id.historyTextView)
        defectRecyclerView = findViewById(R.id.defectRecyclerView)
    }

    private fun setupRecyclerView() {
        defectRecyclerView.layoutManager = LinearLayoutManager(this)
        defectRecyclerView.adapter = defectAdapter
    }

    private fun setupBluetooth() {
        bluetoothAdapter = getSystemService(BluetoothManager::class.java)?.adapter
        if (bluetoothAdapter == null) {
            statusTextView.text = "连接状态：本机不支持蓝牙"
        }
    }

    private fun setupButtons() {
        connectButton.setOnClickListener {
            selectedDevice?.let { device ->
                connectToDevice(device)
            } ?: showToast("请先选择一个已配对蓝牙设备")
        }

        startButton.setOnClickListener {
            startReceiving()
        }

        stopButton.setOnClickListener {
            stopReceiving()
            statusTextView.text = if (socket?.isConnected == true) {
                "连接状态：已停止接收，蓝牙仍连接"
            } else {
                "连接状态：未连接"
            }
            showToast("已停止接收")
        }

        clearButton.setOnClickListener {
            historyRecords.clear()
            saveHistory()
            updateHistoryText()
            rawJsonTextView.text = "暂无数据"
            defectAdapter.submitList(emptyList())
            defectStatusTextView.text = "是否检测到缺陷：暂无数据"
            defectCountTextView.text = "缺陷总数：0"
            overallLevelTextView.text = "整体严重程度：暂无数据"
            summaryTextView.text = "各类缺陷数量：暂无数据"
        }

        deviceListView.setOnItemClickListener { _, _, position, _ ->
            selectedDevice = pairedDevices.getOrNull(position)
            selectedDevice?.let {
                statusTextView.text = "连接状态：已选择 ${deviceNames[position]}"
                connectToDevice(it)
            }
        }
    }

    private fun requiredPermissions(): Array<String> {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            arrayOf(
                Manifest.permission.BLUETOOTH_SCAN,
                Manifest.permission.BLUETOOTH_CONNECT,
                Manifest.permission.ACCESS_FINE_LOCATION
            )
        } else {
            arrayOf(Manifest.permission.ACCESS_FINE_LOCATION)
        }
    }

    private fun hasRequiredPermissions(): Boolean {
        return requiredPermissions().all { permission ->
            ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun requestRequiredPermissions() {
        ActivityCompat.requestPermissions(this, requiredPermissions(), REQUEST_PERMISSIONS)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_PERMISSIONS) {
            if (grantResults.isNotEmpty() && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                loadPairedDevices()
            } else {
                statusTextView.text = "连接状态：缺少蓝牙权限"
                showToast("没有蓝牙权限，无法连接设备")
            }
        }
    }

    @Deprecated("Android framework callback")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_ENABLE_BLUETOOTH && hasRequiredPermissions()) {
            loadPairedDevices()
        }
    }

    @SuppressLint("MissingPermission")
    private fun loadPairedDevices() {
        val adapter = bluetoothAdapter
        if (adapter == null) {
            showToast("本机不支持蓝牙")
            return
        }

        if (!hasRequiredPermissions()) {
            requestRequiredPermissions()
            return
        }

        if (!adapter.isEnabled) {
            statusTextView.text = "连接状态：蓝牙未开启"
            startActivityForResult(Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE), REQUEST_ENABLE_BLUETOOTH)
            return
        }

        pairedDevices.clear()
        deviceNames.clear()

        val bonded = adapter.bondedDevices
        if (bonded.isNullOrEmpty()) {
            statusTextView.text = "连接状态：没有已配对设备，请先在系统蓝牙设置中配对 HC-05/HC-06"
        } else {
            pairedDevices.addAll(bonded)
            for (device in pairedDevices) {
                val name = device.name ?: "未知设备"
                val address = device.address ?: "未知地址"
                deviceNames.add("$name\n$address")
            }
            selectedDevice = pairedDevices.firstOrNull()
            statusTextView.text = "连接状态：请选择蓝牙设备"
        }

        val adapterView = ArrayAdapter(this, android.R.layout.simple_list_item_single_choice, deviceNames)
        deviceListView.adapter = adapterView
        if (pairedDevices.isNotEmpty()) {
            deviceListView.setItemChecked(0, true)
        }
    }

    @SuppressLint("MissingPermission")
    private fun connectToDevice(device: BluetoothDevice) {
        if (!hasRequiredPermissions()) {
            requestRequiredPermissions()
            return
        }

        val adapter = bluetoothAdapter
        if (adapter == null) {
            showToast("本机不支持蓝牙")
            return
        }

        if (!adapter.isEnabled) {
            showToast("请先开启蓝牙")
            startActivityForResult(Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE), REQUEST_ENABLE_BLUETOOTH)
            return
        }

        statusTextView.text = "连接状态：正在连接 ${device.name ?: device.address}"
        stopReceiving()
        closeSocket()

        thread(name = "BluetoothConnectThread") {
            try {
                adapter.cancelDiscovery()
                val newSocket = device.createRfcommSocketToServiceRecord(SPP_UUID)
                newSocket.connect()
                socket = newSocket
                inputStream = newSocket.inputStream
                runOnUiThread {
                    statusTextView.text = "连接状态：已连接 ${device.name ?: device.address}"
                    showToast("蓝牙连接成功")
                }
            } catch (e: Exception) {
                closeSocket()
                runOnUiThread {
                    statusTextView.text = "连接状态：连接失败"
                    showToast("连接失败：${e.message ?: "未知错误"}")
                }
            }
        }
    }

    private fun startReceiving() {
        if (receiving) {
            showToast("正在接收数据")
            return
        }

        val stream = inputStream
        val btSocket = socket
        if (stream == null || btSocket == null || !btSocket.isConnected) {
            showToast("请先连接蓝牙设备")
            return
        }

        receiving = true
        statusTextView.text = "连接状态：正在接收数据"

        thread(name = "BluetoothReadThread") {
            val lineBuffer = StringBuilder()
            val buffer = ByteArray(256)

            try {
                while (receiving) {
                    val available = stream.available()
                    if (available <= 0) {
                        Thread.sleep(40)
                        continue
                    }

                    val count = stream.read(buffer, 0, minOf(buffer.size, available))
                    if (count < 0) {
                        throw IOException("蓝牙设备已断开")
                    }

                    val text = String(buffer, 0, count, Charsets.UTF_8)
                    for (ch in text) {
                        when (ch) {
                            '\n' -> {
                                val line = lineBuffer.toString().trim()
                                lineBuffer.clear()
                                if (line.isNotEmpty()) {
                                    handleJsonLine(line)
                                }
                            }
                            '\r' -> Unit
                            else -> lineBuffer.append(ch)
                        }
                    }
                }
            } catch (e: Exception) {
                val message = e.message ?: "读取失败"
                receiving = false
                runOnUiThread {
                    statusTextView.text = "连接状态：设备断开或读取失败"
                    showToast(message)
                }
            } finally {
                receiving = false
            }
        }
    }

    private fun stopReceiving() {
        receiving = false
    }

    private fun handleJsonLine(line: String) {
        try {
            val result = DetectionResult.fromJson(line)
            val record = buildHistoryRecord(result)
            historyRecords.add(0, record)
            while (historyRecords.size > MAX_HISTORY) {
                historyRecords.removeAt(historyRecords.lastIndex)
            }
            saveHistory()

            runOnUiThread {
                updateDetectionUi(result, line)
                updateHistoryText()
            }
        } catch (e: Exception) {
            runOnUiThread {
                rawJsonTextView.text = line
                showToast("JSON 解析失败：${e.message ?: "格式错误"}")
            }
        }
    }

    private fun updateDetectionUi(result: DetectionResult, rawJson: String) {
        defectStatusTextView.text = if (result.has_defect) {
            "是否检测到缺陷：是"
        } else {
            "是否检测到缺陷：否"
        }
        defectCountTextView.text = "缺陷总数：${result.defect_count}"
        overallLevelTextView.text = "整体严重程度：${levelName(result.overall_level)}"
        summaryTextView.text = buildSummaryText(result)
        rawJsonTextView.text = rawJson
        defectAdapter.submitList(result.defects)
    }

    private fun buildSummaryText(result: DetectionResult): String {
        return buildString {
            append("各类缺陷数量：\n")
            for (type in DEFECT_TYPE_ORDER) {
                append(defectTypeName(type))
                append("：")
                append(result.summary[type] ?: 0)
                append('\n')
            }
        }.trimEnd()
    }

    private fun buildHistoryRecord(result: DetectionResult): String {
        val timeText = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.CHINA).format(Date())
        val status = if (result.has_defect) "发现缺陷" else "正常"
        return "$timeText | $status | 数量 ${result.defect_count} | 等级 ${levelName(result.overall_level)} | OpenMV时间戳 ${result.timestamp}"
    }

    private fun loadHistory() {
        historyRecords.clear()
        val raw = getSharedPreferences(HISTORY_PREF, MODE_PRIVATE).getString(HISTORY_KEY, null)
        if (!raw.isNullOrBlank()) {
            try {
                val array = JSONArray(raw)
                for (i in 0 until array.length()) {
                    historyRecords.add(array.optString(i))
                }
            } catch (_: Exception) {
                historyRecords.clear()
            }
        }
        updateHistoryText()
    }

    private fun saveHistory() {
        val array = JSONArray()
        for (record in historyRecords) {
            array.put(record)
        }
        getSharedPreferences(HISTORY_PREF, MODE_PRIVATE)
            .edit()
            .putString(HISTORY_KEY, array.toString())
            .apply()
    }

    private fun updateHistoryText() {
        historyTextView.text = if (historyRecords.isEmpty()) {
            "暂无记录"
        } else {
            historyRecords.joinToString(separator = "\n")
        }
    }

    private fun closeSocket() {
        try {
            inputStream?.close()
        } catch (_: Exception) {
        }
        try {
            socket?.close()
        } catch (_: Exception) {
        }
        inputStream = null
        socket = null
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }
}
