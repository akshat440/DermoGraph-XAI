import { useState } from "react"
import { CheckCircle, AlertTriangle, AlertCircle, Download, Eye, BarChart3, Info, Zap, Shield } from "lucide-react"

const RISK_CONFIG = {
  HIGH:   { icon: AlertTriangle, color: "text-red-400",   bg: "bg-red-500/10",   border: "border-red-500/20",   label: "High Risk" },
  MEDIUM: { icon: AlertCircle,  color: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/20", label: "Medium Risk" },
  LOW:    { icon: CheckCircle,  color: "text-teal-400",  bg: "bg-teal-500/10",  border: "border-teal-500/20",  label: "Low Risk" },
}

const CLASS_COLORS = {
  "Melanoma":             "#ef4444",
  "Nevi":                 "#22c55e",
  "Basal Cell Carcinoma": "#f97316",
  "Actinic Keratosis":    "#eab308",
  "Benign Keratosis":     "#22c55e",
  "Dermatofibroma":       "#22c55e",
  "Vascular Lesion":      "#3b82f6",
}

const RARE_CLASSES = ["Vascular Lesion", "Dermatofibroma"]

const CLASS_SUPPORT = {
  "Melanoma":             636,
  "Nevi":                 2469,
  "Basal Cell Carcinoma": 133,
  "Actinic Keratosis":    125,
  "Benign Keratosis":     133,
  "Dermatofibroma":       9,
  "Vascular Lesion":      12,
}

function ProbBar({ label, value, color, isTop }) {
  return (
    <div className={`space-y-1.5 ${isTop ? "opacity-100" : "opacity-55"}`}>
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: color }} />
          <span className={`text-xs font-medium ${isTop ? "text-white" : "text-slate-400"}`}>{label}</span>
          {RARE_CLASSES.includes(label) && (
            <span className="text-[9px] px-1 py-0.5 rounded bg-blue-500/15 text-blue-400 border border-blue-500/20 font-medium">
              RARE
            </span>
          )}
        </div>
        <span className={`text-xs font-bold tabular-nums ${isTop ? "text-white" : "text-slate-500"}`}>
          {value.toFixed(1)}%
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{ width: `${Math.min(value, 100)}%`, backgroundColor: color }}
        />
      </div>
    </div>
  )
}

function WarnBanner({ icon: Icon, color, bg, border, children }) {
  return (
    <div className={`flex items-start gap-2.5 p-3 rounded-xl border ${bg} ${border}`}>
      <Icon size={13} className={`${color} shrink-0 mt-0.5`} />
      <p className={`text-xs leading-relaxed ${color}`}>{children}</p>
    </div>
  )
}

export default function ResultsPanel({ result, preview, ensemble }) {
  const [tab, setTab] = useState("result")

  const risk       = result.class_info?.risk || "LOW"
  const riskConfig = RISK_CONFIG[risk]
  const RiskIcon   = riskConfig.icon
  const confidence = result.confidence || 0
  const isLowConf  = confidence < 60
  const isMedConf  = confidence >= 60 && confidence < 75
  const isRare     = RARE_CLASSES.includes(result.predicted_class)
  const isHighRisk = risk === "HIGH"

  const probs = result.probabilities
    ? Object.entries(result.probabilities).sort(([, a], [, b]) => b - a)
    : []

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement("a")
    a.href     = url
    a.download = `dermograph_${(result.predicted_class || "result").replace(/\s+/g, "_")}_${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/50 overflow-hidden flex flex-col h-full">

      {/* Header */}
      <div className={`p-5 border-b border-slate-800 ${riskConfig.bg}`}>
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold
              ${riskConfig.bg} ${riskConfig.border} border ${riskConfig.color} mb-2`}>
              <RiskIcon size={10} />
              {riskConfig.label}
            </div>
            <h2 className="text-2xl font-black text-white tracking-tight">{result.predicted_class}</h2>
            <p className="text-slate-400 text-xs mt-1 leading-relaxed line-clamp-2">{result.class_info?.description}</p>
          </div>
          <div className="text-right shrink-0">
            <div className={`text-3xl font-black tabular-nums
              ${isLowConf ? "text-red-400" : isMedConf ? "text-amber-400" : "text-white"}`}>
              {confidence.toFixed(1)}%
            </div>
            <div className="text-xs text-slate-500">confidence</div>
          </div>
        </div>

        <div className="flex items-center gap-3 mt-3 text-xs text-slate-500 flex-wrap">
          <span>Model: <span className="text-slate-300">{result.model_name || "Ensemble"}</span></span>
          {result.class_info?.icd && <span>ICD: <span className="text-slate-300">{result.class_info.icd}</span></span>}
          {result.inference_ms && <span>{result.inference_ms}ms</span>}
          {ensemble && <span className="text-teal-400 flex items-center gap-1"><Shield size={10} /> Ensemble</span>}
        </div>

        {/* Warnings */}
        <div className="mt-3 space-y-2">
          {isLowConf && (
            <WarnBanner icon={AlertTriangle} color="text-red-300" bg="bg-red-500/10" border="border-red-500/20">
              <strong>Low confidence ({confidence.toFixed(1)}%)</strong> — Model is uncertain. Try{" "}
              <strong>Ensemble mode</strong> or consult a specialist.
            </WarnBanner>
          )}
          {isMedConf && (
            <WarnBanner icon={AlertCircle} color="text-amber-300" bg="bg-amber-500/10" border="border-amber-500/20">
              <strong>Moderate confidence ({confidence.toFixed(1)}%)</strong> — Consider Ensemble mode for better reliability.
            </WarnBanner>
          )}
          {isRare && (
            <WarnBanner icon={Info} color="text-blue-300" bg="bg-blue-500/10" border="border-blue-500/20">
              <strong>Rare class detected</strong> — {result.predicted_class} has only{" "}
              {CLASS_SUPPORT[result.predicted_class]} test samples. All models are less reliable for this class.
              Use <strong>Ensemble mode</strong> and verify with a dermatologist.
            </WarnBanner>
          )}
          {isHighRisk && confidence >= 75 && (
            <WarnBanner icon={AlertTriangle} color="text-red-300" bg="bg-red-500/10" border="border-red-500/20">
              <strong>High risk lesion detected.</strong> Please consult a qualified dermatologist immediately.
            </WarnBanner>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-800">
        {[
          { id: "result",  label: "Results", icon: BarChart3 },
          { id: "heatmap", label: "Heatmap", icon: Eye },
          { id: "abcde",   label: "ABCDE",   icon: Info },
        ].map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`flex items-center gap-1.5 px-4 py-3 text-xs font-medium transition-colors flex-1 justify-center
              ${tab === id ? "text-teal-400 border-b-2 border-teal-400 bg-teal-500/5" : "text-slate-500 hover:text-slate-300"}`}
          >
            <Icon size={12} />
            {label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="p-5 flex-1 overflow-y-auto">

        {tab === "result" && (
          <div className="space-y-3">
            <div className="flex items-center justify-between mb-1">
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Probability Distribution</h3>
            </div>

            {probs.map(([cls, prob], i) => (
              <ProbBar key={cls} label={cls} value={prob} color={CLASS_COLORS[cls] || "#6b7280"} isTop={i === 0} />
            ))}

            {/* Training support context */}
            <div className="mt-4 pt-4 border-t border-slate-800">
              <p className="text-xs text-slate-500 mb-2 font-medium">Training reliability (test samples):</p>
              <div className="grid grid-cols-2 gap-1.5">
                {Object.entries(CLASS_SUPPORT).map(([cls, n]) => (
                  <div key={cls} className="flex items-center justify-between px-2 py-1.5 rounded-lg bg-slate-800/40">
                    <span className="text-xs text-slate-500 truncate mr-2">{cls}</span>
                    <span className={`text-xs font-bold tabular-nums shrink-0
                      ${n < 20 ? "text-red-400" : n < 100 ? "text-amber-400" : "text-teal-400"}`}>
                      n={n}
                    </span>
                  </div>
                ))}
              </div>
              <p className="text-xs text-slate-600 mt-1.5 italic">
                🔴 &lt;20 = unreliable · 🟡 &lt;100 = moderate · 🟢 100+ = reliable
              </p>
            </div>

            {/* Ensemble suggestion */}
            {!ensemble && (isLowConf || isMedConf || isRare) && (
              <div className="p-3 rounded-xl bg-teal-500/5 border border-teal-500/15 flex items-start gap-2 mt-1">
                <Zap size={13} className="text-teal-400 shrink-0 mt-0.5" />
                <p className="text-xs text-teal-300 leading-relaxed">
                  <strong>Try Ensemble mode</strong> — averages MaxViT-T + EfficientNet-B3 for more reliable predictions
                  on uncertain or rare classes.
                </p>
              </div>
            )}

            {/* Stats */}
            <div className="grid grid-cols-3 gap-3 pt-3 border-t border-slate-800 mt-1">
              {[
                ["Inference", result.inference_ms ? `${result.inference_ms}ms` : "—"],
                ["Image",     result.image_size || "224×224"],
                ["Time",      result.timestamp ? new Date(result.timestamp).toLocaleTimeString() : "—"],
              ].map(([label, value]) => (
                <div key={label} className="text-center">
                  <div className="text-white text-sm font-bold">{value}</div>
                  <div className="text-slate-500 text-xs">{label}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {tab === "heatmap" && (
          <div className="space-y-4">
            {result.gradcam_image ? (
              <>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <p className="text-xs text-slate-500 mb-2 text-center font-medium">Original</p>
                    <img src={preview} alt="Original" className="w-full rounded-xl object-cover aspect-square border border-slate-700" />
                  </div>
                  <div>
                    <p className="text-xs text-slate-500 mb-2 text-center font-medium">GradCAM Heatmap</p>
                    <img src={result.gradcam_image} alt="GradCAM" className="w-full rounded-xl object-cover aspect-square border border-slate-700" />
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-xs text-slate-500 shrink-0">Low focus</span>
                  <div className="flex-1 h-2 rounded-full" style={{ background: "linear-gradient(to right, #00008b, #0000ff, #00ffff, #00ff00, #ffff00, #ff8000, #ff0000)" }} />
                  <span className="text-xs text-slate-500 shrink-0">High focus</span>
                </div>
                <div className="flex items-start gap-2 p-3 rounded-lg bg-slate-800/50">
                  <Info size={14} className="text-teal-400 shrink-0 mt-0.5" />
                  <p className="text-xs text-slate-400 leading-relaxed">
                    <strong className="text-white">Red/warm areas</strong> show model focus regions.
                    Verify the model is focusing on the actual lesion, not background artifacts like hair or rulers.
                  </p>
                </div>
              </>
            ) : (
              <div className="text-center py-12">
                <Eye size={32} className="text-slate-700 mx-auto mb-3" />
                <p className="text-slate-400 text-sm font-medium">GradCAM not generated</p>
                <p className="text-slate-600 text-xs mt-1">Enable GradCAM toggle in Options before analyzing</p>
              </div>
            )}
          </div>
        )}

        {tab === "abcde" && (
          <div className="space-y-3">
            {[
              ["A", "Asymmetry", result.class_info?.abcde?.[0] || "—"],
              ["B", "Border",    result.class_info?.abcde?.[1] || "—"],
              ["C", "Color",     result.class_info?.abcde?.[2] || "—"],
              ["D", "Diameter",  result.class_info?.abcde?.[3] || "—"],
              ["E", "Evolution", result.class_info?.abcde?.[4] || "—"],
            ].map(([letter, criterion, value]) => (
              <div key={letter} className="flex items-center gap-3 p-3 rounded-xl bg-slate-800/50 border border-slate-700/30">
                <div className="w-9 h-9 rounded-lg bg-teal-500/15 border border-teal-500/25 flex items-center justify-center shrink-0">
                  <span className="text-teal-400 font-black text-sm">{letter}</span>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-xs text-slate-500 font-medium">{criterion}</div>
                  <div className="text-sm text-white font-semibold mt-0.5">{value}</div>
                </div>
              </div>
            ))}
            <div className="p-3 rounded-xl bg-amber-500/5 border border-amber-500/15 mt-1">
              <div className="flex items-center gap-2 mb-1.5">
                <Info size={12} className="text-amber-400" />
                <span className="text-xs font-semibold text-amber-300">Innovation Module — Coming Soon</span>
              </div>
              <p className="text-xs text-amber-200/60 leading-relaxed">
                Full quantitative ABCDE scoring using the 7-point checklist (Derm7pt dataset) is in development.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="px-5 pb-5 pt-3 border-t border-slate-800 flex gap-2 shrink-0">
        <button
          onClick={handleExport}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white text-xs font-medium transition-colors"
        >
          <Download size={13} />
          Export JSON
        </button>
        <button disabled className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-800/40 border border-slate-700/40 text-slate-600 text-xs font-medium cursor-not-allowed">
          <Download size={13} />
          PDF Report
          <span className="text-teal-400/60 text-[10px] ml-0.5">Soon</span>
        </button>
      </div>
    </div>
  )
}
