import { ExternalLink, BookOpen, Database, Github } from "lucide-react"

export default function ResearchPage() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-10">
      <div className="mb-8">
        <h1 className="text-3xl font-black text-white mb-2">Research</h1>
        <p className="text-slate-400">DermoGraph-XAI — Team 8, VIT Bhopal</p>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 mb-6">
        <h2 className="text-lg font-bold text-white mb-3 flex items-center gap-2">
          <BookOpen size={16} className="text-teal-400" /> About
        </h2>
        <p className="text-slate-300 text-sm leading-relaxed mb-3">
          DermoGraph-XAI is a comprehensive deep learning framework for automated skin lesion
          classification across 7 diagnostic categories. The system benchmarks 12+ CNN and
          transformer architectures on a unified 6-dataset corpus of 35,084 dermoscopy images.
        </p>
        <p className="text-slate-300 text-sm leading-relaxed">
          The project introduces DermoNet — a novel hybrid architecture combining Dual-Scale CNN Stems,
          Lesion-Aware Attention Gates (LAAG), and Multi-Resolution Transformer Blocks (MRTB).
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
        {[
          { icon: Github,   label: "GitHub Repository", url: "https://github.com/akshat440/DermoGraph-XAI", sub: "Code + Notebooks" },
          { icon: Database, label: "Kaggle Datasets",    url: "https://www.kaggle.com/akshat23029",           sub: "All 6 datasets public" },
        ].map(link => (
          <a key={link.label} href={link.url} target="_blank" rel="noopener noreferrer"
            className="flex items-center gap-4 p-4 rounded-xl border border-slate-700 bg-slate-800/30 hover:border-teal-500/30 hover:bg-teal-500/5 transition-all group">
            <div className="w-10 h-10 rounded-xl bg-slate-700 flex items-center justify-center shrink-0 group-hover:bg-teal-500/20 transition-colors">
              <link.icon size={18} className="text-slate-300 group-hover:text-teal-400 transition-colors" />
            </div>
            <div className="flex-1">
              <div className="text-white font-semibold text-sm">{link.label}</div>
              <div className="text-slate-500 text-xs">{link.sub}</div>
            </div>
            <ExternalLink size={14} className="text-slate-600 group-hover:text-teal-400 transition-colors" />
          </a>
        ))}
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 mb-6">
        <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <Database size={16} className="text-teal-400" /> Dataset Citations
        </h2>
        <div className="space-y-3">
          {[
            { name: "HAM10000",        cite: "Tschandl et al., Scientific Data 2018",     doi: "10.1038/sdata.2018.161",         license: "CC BY-NC-SA 4.0" },
            { name: "ISIC 2020",       cite: "Rotemberg et al., Scientific Data 2021",     doi: "10.1038/s41597-021-00815-z",     license: "CC BY-NC-SA 4.0" },
            { name: "PAD-UFES-20",     cite: "Pacheco et al., Data in Brief 2020",         doi: "10.1016/j.dib.2020.106221",      license: "CC BY 4.0" },
            { name: "FitzPatrick17k",  cite: "Groh et al., CVPR Workshop 2021",            doi: "github.com/mattgroh/fitzpatrick17k", license: "MIT" },
            { name: "Derm7pt",         cite: "Kawahara et al., IEEE JBHI 2019",            doi: "10.1109/JBHI.2018.2824327",      license: "See source" },
            { name: "MIDAS",           cite: "Kaggle Community Dataset",                    doi: "kaggle.com",                     license: "See source" },
            { name: "Melanoma Cancer", cite: "SIIM-ISIC Challenge 2020",                    doi: "kaggle.com/c/siim-isic",         license: "CC BY-NC-SA 4.0" },
          ].map(ds => (
            <div key={ds.name} className="flex items-start gap-3 p-3 rounded-lg bg-slate-800/30">
              <div className="w-2 h-2 rounded-full bg-teal-400/60 shrink-0 mt-1.5" />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-white text-sm font-semibold">{ds.name}</span>
                  <span className="text-xs px-1.5 py-0.5 rounded bg-slate-700 text-slate-400">{ds.license}</span>
                </div>
                <div className="text-xs text-slate-400 mt-0.5">{ds.cite}</div>
                <div className="text-xs text-slate-600 font-mono mt-0.5">{ds.doi}</div>
              </div>
            </div>
          ))}
        </div>
        <p className="text-xs text-slate-600 mt-4 italic">All datasets used strictly for academic research purposes.</p>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
        <h2 className="text-lg font-bold text-white mb-3">Team</h2>
        <p className="text-slate-400 text-sm">Team 8 — VIT Bhopal</p>
        <p className="text-slate-500 text-sm">B.Tech Final Year Project · Department of Computer Science</p>
        <p className="text-xs text-slate-600 mt-3 italic">
          For research purposes only. Not a substitute for professional medical diagnosis.
        </p>
      </div>
    </div>
  )
}
