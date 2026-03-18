import { useState, useEffect } from "react"
import { Cpu, CheckCircle, XCircle } from "lucide-react"

const API = "http://localhost:8000"

export default function ModelsPage() {
  const [models, setModels] = useState([])

  useEffect(() => {
    fetch(`${API}/models`)
      .then(r => r.json())
      .then(d => setModels(d.models || []))
      .catch(() => {})
  }, [])

  return (
    <div className="max-w-7xl mx-auto px-6 py-10">
      <div className="mb-8">
        <h1 className="text-3xl font-black text-white mb-2">Models</h1>
        <p className="text-slate-400">All trained models available for inference</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {models.map(m => (
          <div key={m.key} className={`rounded-2xl border p-5 transition-all
            ${m.loaded ? "border-slate-700 bg-slate-900/50" : "border-slate-800 bg-slate-900/20 opacity-60"}`}>
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-slate-800 flex items-center justify-center">
                  <Cpu size={18} className="text-slate-400" />
                </div>
                <div>
                  <h3 className="text-white font-bold">{m.name}</h3>
                  <span className={`text-xs px-2 py-0.5 rounded font-medium
                    ${m.type === "Transformer" ? "bg-purple-500/20 text-purple-400" : "bg-slate-700 text-slate-400"}`}>
                    {m.type}
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-1.5">
                {m.loaded
                  ? <><CheckCircle size={14} className="text-teal-400" /><span className="text-xs text-teal-400">Loaded</span></>
                  : <><XCircle size={14} className="text-slate-500" /><span className="text-xs text-slate-500">Not loaded</span></>
                }
              </div>
            </div>
            <p className="text-sm text-slate-400 mb-4">{m.description}</p>
            <div className="grid grid-cols-3 gap-3">
              {[["Accuracy", `${m.accuracy}%`], ["F1 Macro", m.f1], ["AUC-ROC", m.auc]].map(([label, value]) => (
                <div key={label} className="text-center p-2 rounded-lg bg-slate-800/50">
                  <div className="text-white font-bold text-sm">{value}</div>
                  <div className="text-slate-500 text-xs">{label}</div>
                </div>
              ))}
            </div>
            <div className="mt-3 flex items-center justify-between text-xs text-slate-500">
              <span>{m.params} parameters</span>
              <span className="font-mono text-slate-600">{m.key}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Innovation modules */}
      <div className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
        <h2 className="text-lg font-bold text-white mb-4">Innovation Modules</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          {[
            { name: "ABCDE Branch",      desc: "7-point checklist feature extraction using Derm7pt dataset",      status: "progress" },
            { name: "GAT Pattern Graph",  desc: "Graph Attention Network for lesion pattern relationships",         status: "planned" },
            { name: "Neural ODE",         desc: "Continuous-depth modeling for lesion evolution tracking",          status: "planned" },
            { name: "Fairness MTL",       desc: "Multi-task learning for skin tone fairness (FitzPatrick I–VI)",    status: "planned" },
          ].map(mod => (
            <div key={mod.name} className={`p-4 rounded-xl border
              ${mod.status === "progress" ? "border-teal-500/20 bg-teal-500/5" : "border-slate-800 bg-slate-800/30"}`}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-white font-semibold text-sm">{mod.name}</span>
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium
                  ${mod.status === "progress" ? "bg-teal-500/20 text-teal-400" : "bg-slate-700 text-slate-500"}`}>
                  {mod.status === "progress" ? "🔄 In Progress" : "📋 Planned"}
                </span>
              </div>
              <p className="text-xs text-slate-500">{mod.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
