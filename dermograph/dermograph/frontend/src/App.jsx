import { useState, useEffect } from "react"
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from "react-router-dom"
import AnalyzePage from "./pages/AnalyzePage"
import DashboardPage from "./pages/DashboardPage"
import ModelsPage from "./pages/ModelsPage"
import ResearchPage from "./pages/ResearchPage"

function NavLink({ to, children }) {
  const location = useLocation()
  const active   = location.pathname === to
  return (
    <Link
      to={to}
      className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200
        ${active ? "bg-teal-500/20 text-teal-400 border border-teal-500/30" : "text-slate-400 hover:text-white hover:bg-white/5"}`}
    >
      {children}
    </Link>
  )
}

function Navbar({ apiStatus, modelsLoaded }) {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-white/5 bg-slate-950/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-teal-400 to-cyan-500 flex items-center justify-center">
            <span className="text-slate-950 font-black text-sm">D</span>
          </div>
          <div>
            <span className="text-white font-bold text-sm tracking-tight">DermoGraph</span>
            <span className="text-teal-400 font-bold text-sm">-XAI</span>
          </div>
        </Link>
        <div className="flex items-center gap-1">
          <NavLink to="/">Analyze</NavLink>
          <NavLink to="/dashboard">Dashboard</NavLink>
          <NavLink to="/models">Models</NavLink>
          <NavLink to="/research">Research</NavLink>
        </div>
        <div className="flex items-center gap-3">
          {apiStatus === "online" && (
            <span className="text-xs text-slate-500">{modelsLoaded} models</span>
          )}
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${apiStatus === "online" ? "bg-teal-400 animate-pulse" : "bg-red-400"}`} />
            <span className="text-xs text-slate-500">{apiStatus === "online" ? "API Online" : "API Offline"}</span>
          </div>
        </div>
      </div>
    </nav>
  )
}

export default function App() {
  const [apiStatus, setApiStatus]     = useState("checking")
  const [modelsLoaded, setModelsLoaded] = useState(0)

  useEffect(() => {
    fetch("http://localhost:8000/health")
      .then(r => r.json())
      .then(d => {
        setApiStatus("online")
        setModelsLoaded(d.models_loaded || 0)
      })
      .catch(() => setApiStatus("offline"))
  }, [])

  return (
    <Router>
      <div className="min-h-screen bg-slate-950 text-white">
        <Navbar apiStatus={apiStatus} modelsLoaded={modelsLoaded} />
        <main className="pt-16">
          <Routes>
            <Route path="/"          element={<AnalyzePage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/models"    element={<ModelsPage />} />
            <Route path="/research"  element={<ResearchPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}
