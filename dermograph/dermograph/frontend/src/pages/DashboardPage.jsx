import { useState, useEffect } from "react"
import { TrendingUp, Award, Zap, BarChart2, Target, Activity } from "lucide-react"

const API = "http://localhost:8000"

function StatCard({ icon: Icon, label, value, sub, color = "teal" }) {
  const c = {
    teal:   "from-teal-500/20 to-teal-500/5 border-teal-500/20 text-teal-400",
    cyan:   "from-cyan-500/20 to-cyan-500/5 border-cyan-500/20 text-cyan-400",
    blue:   "from-blue-500/20 to-blue-500/5 border-blue-500/20 text-blue-400",
    purple: "from-purple-500/20 to-purple-500/5 border-purple-500/20 text-purple-400",
  }
  return (
    <div className={`rounded-2xl border bg-gradient-to-br p-5 ${c[color]}`}>
      <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center mb-3">
        <Icon size={18} className={c[color].split(" ").pop()} />
      </div>
      <div className="text-2xl font-black text-white mb-0.5">{value}</div>
      <div className="text-sm font-medium text-white/80">{label}</div>
      {sub && <div className="text-xs text-white/40 mt-1">{sub}</div>}
    </div>
  )
}

function ModelRow({ model, isTop }) {
  return (
    <div className={`flex items-center gap-4 p-4 rounded-xl transition-colors
      ${isTop ? "bg-teal-500/5 border border-teal-500/15" : "hover:bg-slate-800/30"}`}>
      <div className="w-8 text-center">
        {isTop
          ? <Award size={16} className="text-teal-400 mx-auto" />
          : <span className="text-xs text-slate-600 font-mono">#{model.rank}</span>
        }
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className={`text-sm font-semibold ${isTop ? "text-teal-300" : "text-white"}`}>{model.model}</span>
          <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium
            ${model.type === "Transformer" ? "bg-purple-500/20 text-purple-400" : "bg-slate-700 text-slate-400"}`}>
            {model.type}
          </span>
        </div>
        <div className="flex items-center gap-1 mt-1.5">
          <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-700 ${isTop ? "bg-teal-400" : "bg-slate-600"}`}
              style={{ width: `${(model.accuracy / 91.98) * 100}%` }}
            />
          </div>
        </div>
      </div>
      <div className="text-right shrink-0">
        <div className={`text-sm font-bold ${isTop ? "text-teal-300" : "text-white"}`}>{model.accuracy}%</div>
        <div className="text-xs text-slate-500">F1: {model.f1}</div>
      </div>
      <div className="text-right shrink-0 w-20">
        <div className="text-xs text-slate-400">{model.auc}</div>
        <div className="text-xs text-slate-600">AUC</div>
      </div>
      <div className="text-right shrink-0 w-16">
        <div className="text-xs text-slate-500">{model.params}</div>
      </div>
    </div>
  )
}

export default function DashboardPage() {
  const [benchmark, setBenchmark] = useState([])

  useEffect(() => {
    fetch(`${API}/benchmark`)
      .then(r => r.json())
      .then(d => setBenchmark(d.benchmark || []))
      .catch(() => {})
  }, [])

  const withRank = [...benchmark]
    .sort((a, b) => b.accuracy - a.accuracy)
    .map((m, i) => ({ ...m, rank: i + 1 }))

  return (
    <div className="max-w-7xl mx-auto px-6 py-10">
      <div className="mb-8">
        <h1 className="text-3xl font-black text-white mb-2">Model Dashboard</h1>
        <p className="text-slate-400">Benchmark results across all trained models on 35,084 dermoscopy images</p>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard icon={Award}    label="Best Accuracy"  value="91.98%"  sub="MaxViT-T"          color="teal" />
        <StatCard icon={Activity} label="Best AUC-ROC"   value="0.9866"  sub="DenseNet121"        color="cyan" />
        <StatCard icon={Zap}      label="Best F1 Macro"  value="0.8325"  sub="MaxViT-T"           color="blue" />
        <StatCard icon={Target}   label="Models Trained" value="7+"      sub="CNN + Transformer"  color="purple" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-8">
        {[
          ["Total Images", "35,084", "6 datasets combined"],
          ["Training Set",  "28,056", "80% of total"],
          ["Test Set",       "3,517", "Held-out evaluation"],
        ].map(([label, value, sub]) => (
          <div key={label} className="rounded-xl border border-slate-800 bg-slate-900/50 p-4 text-center">
            <div className="text-2xl font-black text-white">{value}</div>
            <div className="text-sm text-slate-300 font-medium">{label}</div>
            <div className="text-xs text-slate-500 mt-0.5">{sub}</div>
          </div>
        ))}
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-900/50 overflow-hidden">
        <div className="flex items-center justify-between p-5 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <BarChart2 size={16} className="text-teal-400" />
            <h2 className="text-base font-bold text-white">Benchmark Comparison</h2>
          </div>
          <span className="text-xs text-slate-500">{withRank.length} models</span>
        </div>
        <div className="flex items-center gap-4 px-4 py-2 border-b border-slate-800/50">
          <div className="w-8" />
          <div className="flex-1 text-xs text-slate-500 font-medium uppercase tracking-wider">Model</div>
          <div className="text-xs text-slate-500 font-medium uppercase tracking-wider w-20 text-right">Accuracy</div>
          <div className="text-xs text-slate-500 font-medium uppercase tracking-wider w-20 text-right">AUC</div>
          <div className="text-xs text-slate-500 font-medium uppercase tracking-wider w-16 text-right">Params</div>
        </div>
        <div className="p-3 space-y-1">
          {withRank.map((m, i) => <ModelRow key={m.model} model={m} isTop={i === 0} />)}
          {benchmark.length === 0 && (
            <div className="text-center py-8 text-slate-500 text-sm">Loading...</div>
          )}
        </div>
      </div>

      <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
        <h2 className="text-base font-bold text-white mb-4 flex items-center gap-2">
          <TrendingUp size={16} className="text-teal-400" />
          Dataset Composition
        </h2>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
          {[
            { name: "HAM10000",        n: 10015, color: "#14b8a6" },
            { name: "Melanoma Cancer", n: 10605, color: "#06b6d4" },
            { name: "ISIC 2020",       n: 8757,  color: "#3b82f6" },
            { name: "MIDAS",           n: 3411,  color: "#8b5cf6" },
            { name: "PAD-UFES-20",     n: 2298,  color: "#ec4899" },
            { name: "Derm7pt",         n: 1011,  color: "#f59e0b" },
          ].map(ds => (
            <div key={ds.name} className="flex items-center gap-3 p-3 rounded-xl bg-slate-800/30">
              <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: ds.color }} />
              <div>
                <div className="text-sm text-white font-medium">{ds.name}</div>
                <div className="text-xs text-slate-500">{ds.n.toLocaleString()} images</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
