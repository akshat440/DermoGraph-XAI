import { useState, useRef, useCallback } from "react"
import { Upload, Zap, AlertTriangle, Info, RotateCcw } from "lucide-react"
import ResultsPanel from "../components/ResultsPanel"
import ModelSelector from "../components/ModelSelector"

const API = "http://localhost:8000"

export default function AnalyzePage() {
  const [image, setImage]       = useState(null)
  const [preview, setPreview]   = useState(null)
  const [result, setResult]     = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const [model, setModel]       = useState("maxvit_t")
  const [gradcam, setGradcam]   = useState(true)
  const [dragOver, setDragOver] = useState(false)
  const [ensemble, setEnsemble] = useState(false)
  const fileRef = useRef()

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith("image/")) return
    setImage(file)
    setResult(null)
    setError(null)
    const reader = new FileReader()
    reader.onload = e => setPreview(e.target.result)
    reader.readAsDataURL(file)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }, [handleFile])

  const handleAnalyze = async () => {
    if (!image) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const formData = new FormData()
      formData.append("file", image)
      const url = ensemble
        ? `${API}/predict/ensemble?models=maxvit_t,efficientnet_b3`
        : `${API}/predict?model_key=${model}&gradcam=${gradcam}`
      const res  = await fetch(url, { method: "POST", body: formData })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || "Prediction failed")
      }
      setResult(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-950">
      {/* Hero */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-teal-950/30 via-slate-950 to-slate-950" />
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[300px] bg-teal-500/5 rounded-full blur-3xl" />
        <div className="relative max-w-7xl mx-auto px-6 pt-16 pb-8">
          <div className="text-center mb-10">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-teal-500/20 bg-teal-500/5 text-teal-400 text-xs font-medium mb-4">
              <Zap size={10} />
              Powered by MaxViT-T · 91.98% Accuracy
            </div>
            <h1 className="text-4xl font-black text-white mb-3 tracking-tight">
              Skin Lesion <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-cyan-400">Analysis</span>
            </h1>
            <p className="text-slate-400 text-base max-w-lg mx-auto">
              Upload a dermoscopy image for AI-powered classification across 7 diagnostic categories with explainable heatmaps.
            </p>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 pb-20">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

          {/* LEFT */}
          <div className="space-y-4">

            {/* Upload zone */}
            <div
              className={`relative rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer
                ${dragOver ? "border-teal-400 bg-teal-500/10" : "border-slate-700 hover:border-slate-500"}
                ${preview ? "border-solid border-slate-700" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => !preview && fileRef.current?.click()}
            >
              {preview ? (
                <div className="relative">
                  <img src={preview} alt="Uploaded" className="w-full h-72 object-cover rounded-2xl" />
                  <div className="absolute inset-0 rounded-2xl bg-gradient-to-t from-slate-950/60 to-transparent" />
                  <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between">
                    <span className="text-white text-sm font-medium truncate max-w-xs">{image?.name}</span>
                    <button
                      onClick={(e) => { e.stopPropagation(); setImage(null); setPreview(null); setResult(null); setError(null) }}
                      className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-slate-800/80 backdrop-blur text-slate-300 hover:text-white text-xs transition-colors"
                    >
                      <RotateCcw size={12} /> Reset
                    </button>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-16 px-8">
                  <div className="w-16 h-16 rounded-2xl bg-slate-800 border border-slate-700 flex items-center justify-center mb-4">
                    <Upload size={24} className="text-slate-400" />
                  </div>
                  <p className="text-white font-semibold mb-1">Drop your image here</p>
                  <p className="text-slate-500 text-sm mb-4">or click to browse</p>
                  <p className="text-slate-600 text-xs">JPG, PNG up to 10MB</p>
                </div>
              )}
            </div>
            <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={e => handleFile(e.target.files[0])} />

            {/* Model selector */}
            <ModelSelector model={model} setModel={setModel} ensemble={ensemble} setEnsemble={setEnsemble} />

            {/* Options */}
            <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
              <h3 className="text-sm font-semibold text-slate-300 mb-3">Options</h3>
              <label className="flex items-center justify-between cursor-pointer">
                <div>
                  <span className="text-sm text-white">GradCAM Heatmap</span>
                  <p className="text-xs text-slate-500">Show where the model is looking</p>
                </div>
                <button
                  onClick={() => setGradcam(!gradcam)}
                  className={`w-11 h-6 rounded-full transition-colors relative ${gradcam ? "bg-teal-500" : "bg-slate-700"}`}
                >
                  <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${gradcam ? "translate-x-6" : "translate-x-1"}`} />
                </button>
              </label>
            </div>

            {/* Disclaimer */}
            <div className="flex gap-3 p-4 rounded-xl bg-amber-500/5 border border-amber-500/20">
              <AlertTriangle size={16} className="text-amber-400 shrink-0 mt-0.5" />
              <p className="text-xs text-amber-200/70 leading-relaxed">
                <strong className="text-amber-300">Research use only.</strong> Not a substitute for professional medical diagnosis. Consult a qualified dermatologist for clinical decisions.
              </p>
            </div>

            {/* Analyze button */}
            <button
              onClick={handleAnalyze}
              disabled={!image || loading}
              className={`w-full py-4 rounded-xl font-bold text-sm transition-all duration-200 flex items-center justify-center gap-2
                ${image && !loading
                  ? "bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 shadow-lg shadow-teal-500/20 hover:-translate-y-0.5"
                  : "bg-slate-800 text-slate-500 cursor-not-allowed"}`}
            >
              {loading ? (
                <><div className="w-4 h-4 border-2 border-slate-950/30 border-t-slate-950 rounded-full animate-spin" />Analyzing...</>
              ) : (
                <><Zap size={16} />{ensemble ? "Ensemble Analyze" : "Analyze Image"}</>
              )}
            </button>

            {error && (
              <div className="flex gap-3 p-4 rounded-xl bg-red-500/10 border border-red-500/20">
                <AlertTriangle size={16} className="text-red-400 shrink-0" />
                <p className="text-sm text-red-300">{error}</p>
              </div>
            )}
          </div>

          {/* RIGHT — Results */}
          <div className="min-h-[500px]">
            {result ? (
              <ResultsPanel result={result} preview={preview} ensemble={ensemble} />
            ) : (
              <div className="h-full min-h-[500px] rounded-2xl border border-slate-800 bg-slate-900/30 flex flex-col items-center justify-center gap-4">
                <div className="w-16 h-16 rounded-2xl bg-slate-800/50 border border-slate-700/50 flex items-center justify-center">
                  <Info size={24} className="text-slate-600" />
                </div>
                <div className="text-center">
                  <p className="text-slate-400 font-medium mb-1">Results will appear here</p>
                  <p className="text-slate-600 text-sm">Upload an image and click Analyze</p>
                </div>
                <div className="grid grid-cols-2 gap-2 mt-2 px-8 w-full max-w-sm">
                  {[
                    ["Melanoma", "#ef4444"],
                    ["Nevi", "#22c55e"],
                    ["Basal Cell Ca.", "#f97316"],
                    ["Actinic Kera.", "#eab308"],
                    ["Benign Kera.", "#22c55e"],
                    ["Dermatofibroma", "#22c55e"],
                    ["Vascular", "#3b82f6"],
                  ].map(([name, color]) => (
                    <div key={name} className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: color }} />
                      <span className="text-xs text-slate-500 truncate">{name}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
