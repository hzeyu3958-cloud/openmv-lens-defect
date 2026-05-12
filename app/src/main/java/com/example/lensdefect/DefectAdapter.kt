package com.example.lensdefect

import android.content.Context
import android.graphics.Typeface
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import java.util.Locale

class DefectAdapter : RecyclerView.Adapter<DefectAdapter.DefectViewHolder>() {

    private val items = mutableListOf<Defect>()

    fun submitList(newItems: List<Defect>) {
        items.clear()
        items.addAll(newItems)
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): DefectViewHolder {
        val context = parent.context
        val container = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(12.dp(context), 10.dp(context), 12.dp(context), 10.dp(context))
            layoutParams = RecyclerView.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        }

        val title = TextView(context).apply {
            textSize = 15f
            typeface = Typeface.DEFAULT_BOLD
        }
        val detail = TextView(context).apply {
            textSize = 13f
        }
        container.addView(title)
        container.addView(detail)
        return DefectViewHolder(container, title, detail)
    }

    override fun onBindViewHolder(holder: DefectViewHolder, position: Int) {
        val defect = items[position]
        holder.title.text = "${position + 1}. ${defectTypeName(defect.type)}  ${levelName(defect.level)}"
        holder.detail.text = String.format(
            Locale.US,
            "置信度 %.2f | 坐标 (%d,%d) | 尺寸 %dx%d | 面积 %d | 长度 %d | 长宽比 %.2f",
            defect.confidence,
            defect.x,
            defect.y,
            defect.w,
            defect.h,
            defect.area,
            defect.length,
            defect.aspect_ratio
        )
    }

    override fun getItemCount(): Int = items.size

    class DefectViewHolder(
        container: LinearLayout,
        val title: TextView,
        val detail: TextView
    ) : RecyclerView.ViewHolder(container)

    private fun Int.dp(context: Context): Int {
        return (this * context.resources.displayMetrics.density + 0.5f).toInt()
    }
}
