import { useState, useEffect } from "react"
import { Cpu, Zap, Star } from "lucide-react"

const API = "http://localhost:8000"

const MODEL_META = {
  maxvit_t:        { badge: "Best",  badgeColor: "text-teal-400 bg-teal-500/10 border-teal-500/20" },
  efficientnet_b3: { badge: "Fast",  badgeColor: "text-cyan-400 bg-cyan-500/10 border-cyan-500/20" },
  efficientnet_b0: { badge: "Light", badgeColor: "text-blue-400 bg-blue-500/10 border-blue-500/20" },
  densenet121:     { badge: "Dense", badgeColor: "text-purple-400 bg-purple-500/10 border-purple-500/20" },
  resnet50:        { badge: "Base",  badgeColor: "text-slate-400 bg-slate-500/10 border-slate-500/20" },
}

export default function ModelSelector({ model, setModel, ensemble, setEnsemble }) {
  const [models, setModels] = useState([])

  useEffect(() => {
    fetch(`${API}/models`)
      .then(r => r.json())
      .then(d => setModels(d.models || []))
      .catch(() => {})
  }, [])

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 overflow-hidden">
      <div className="flex items-center justify-between p-4 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Cpu size={14} className="text-slate-400" />
          <span className="text-sm font-semibold text-slate-300">Model</span>
        </div>
        <label className="flex items-center gap-1.5 cursor-pointer">
          <span className="text-xs text-slate-500">Ensemble</span>
          <button
            onClick={() => setEnsemble(!ensemble)}
            className={`w-9 h-5 rounded-full transition-colors relative ${ensemble ? "bg-teal-500" : "bg-slate-700"}`}
          >
            <div className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-transform ${ensemble ? "translate-x-5" : "translate-x-1"}`} />
          </button>
        </label>
      </div>

      {!ensemble ? (
        <div className="p-3 space-y-1.5">
          {models.filter(m => m.loaded).map(m => {
            const meta   = MODEL_META[m.key] || {}
            const active = model === m.key
            return (
              <button
                key={m.key}
                onClick={() => setModel(m.key)}
                className={`w-full flex items-center gap-3 p-3 rounded-lg transition-all text-left
                  ${active ? "bg-teal-500/10 border border-teal-500/20" : "hover:bg-slate-800/50 border border-transparent"}`}
              >
                <div className={`w-2 h-2 rounded-full shrink-0 ${active ? "bg-teal-400" : "bg-slate-600"}`} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className={`text-sm font-medium ${active ? "text-white" : "text-slate-300"}`}>{m.name}</span>
                    {meta.badge && (
                      <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded border ${meta.badgeColor}`}>
                        {meta.badge}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 mt-0.5">
                    <span className="text-xs text-slate-500">{m.accuracy}% acc</span>
                    <span className="text-xs text-slate-600">{m.params}</span>
                    <span className="text-xs text-slate-600">{m.type}</span>
                  </div>
                </div>
                {active && <Zap size={12} className="text-teal-400 shrink-0" />}
              </button>
            )
          })}
          {models.filter(m => !m.loaded).length > 0 && (
            <p className="text-xs text-slate-600 px-3 py-1 border-t border-slate-800 mt-1 pt-2">
              {models.filter(m => !m.loaded).length} model(s) not loaded
            </p>
          )}
          {models.length === 0 && (
            <p className="text-xs text-slate-500 px-3 py-2 text-center">No models loaded — check API</p>
          )}
        </div>
      ) : (
        <div className="p-4">
          <div className="flex items-center gap-2 p-3 rounded-lg bg-teal-500/5 border border-teal-500/15">
            <Star size={14} className="text-teal-400 shrink-0" />
            <div>
              <p className="text-sm text-white font-medium">MaxViT-T + EfficientNet-B3</p>
              <p className="text-xs text-slate-500 mt-0.5">Averages both models · More reliable for uncertain cases</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
