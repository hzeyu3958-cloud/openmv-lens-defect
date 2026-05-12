package com.example.lensdefect

data class Defect(
    val type: String,
    val confidence: Double,
    val x: Int,
    val y: Int,
    val w: Int,
    val h: Int,
    val area: Int,
    val length: Int,
    val aspect_ratio: Double,
    val level: String
)

val DEFECT_TYPE_ORDER = listOf(
    "scratch",
    "dust",
    "stain",
    "coating_damage",
    "crack",
    "edge_damage",
    "unknown"
)

fun defectTypeName(type: String): String {
    return when (type) {
        "scratch" -> "划痕"
        "dust" -> "灰尘颗粒"
        "stain" -> "污点/油污"
        "coating_damage" -> "镀膜损伤"
        "crack" -> "裂纹"
        "edge_damage" -> "边缘损伤"
        "unknown" -> "未知缺陷"
        else -> type
    }
}

fun levelName(level: String): String {
    return when (level) {
        "normal" -> "正常"
        "light" -> "轻微"
        "medium" -> "中等"
        "serious" -> "严重"
        else -> level
    }
}
