package com.example.lensdefect

import org.json.JSONObject

data class DetectionResult(
    val has_defect: Boolean,
    val defect_count: Int,
    val summary: Map<String, Int>,
    val overall_level: String,
    val defects: List<Defect>,
    val timestamp: Long
) {
    companion object {
        fun fromJson(raw: String): DetectionResult {
            val json = JSONObject(raw)
            val summaryJson = json.optJSONObject("summary")
            val summaryMap = linkedMapOf<String, Int>()
            for (type in DEFECT_TYPE_ORDER) {
                summaryMap[type] = summaryJson?.optInt(type, 0) ?: 0
            }

            val defectArray = json.optJSONArray("defects")
            val defectList = mutableListOf<Defect>()
            if (defectArray != null) {
                for (i in 0 until defectArray.length()) {
                    val item = defectArray.getJSONObject(i)
                    defectList.add(
                        Defect(
                            type = item.optString("type", "unknown"),
                            confidence = item.optDouble("confidence", 0.0),
                            x = item.optInt("x", 0),
                            y = item.optInt("y", 0),
                            w = item.optInt("w", 0),
                            h = item.optInt("h", 0),
                            area = item.optInt("area", 0),
                            length = item.optInt("length", 0),
                            aspect_ratio = item.optDouble("aspect_ratio", 0.0),
                            level = item.optString("level", "normal")
                        )
                    )
                }
            }

            return DetectionResult(
                has_defect = json.optBoolean("has_defect", defectList.isNotEmpty()),
                defect_count = json.optInt("defect_count", defectList.size),
                summary = summaryMap,
                overall_level = json.optString("overall_level", "normal"),
                defects = defectList,
                timestamp = json.optLong("timestamp", 0L)
            )
        }
    }
}
