import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import { LivePage } from './pages/LivePage'
import { SessionsPage } from './pages/SessionsPage'
import { SessionPage } from './pages/SessionPage'
import { DspInspectorPage } from './pages/DspInspectorPage'
import { BreathingPage } from './pages/BreathingPage'
import { ExperimentsPage } from './pages/ExperimentsPage'
import './App.css'

export default function App() {
  return (
    <BrowserRouter>
      <div className="layout">
        <nav className="sidebar">
          <div className="brand">csi-sensing</div>
          <NavLink to="/" end>Live</NavLink>
          <NavLink to="/sessions">Sessions</NavLink>
          <NavLink to="/dsp">DSP inspector</NavLink>
          <NavLink to="/breathing">Breathing</NavLink>
          <NavLink to="/experiments">Experiments</NavLink>
        </nav>
        <main className="content">
          <Routes>
            <Route path="/" element={<LivePage />} />
            <Route path="/sessions" element={<SessionsPage />} />
            <Route path="/sessions/:id" element={<SessionPage />} />
            <Route path="/dsp" element={<DspInspectorPage />} />
            <Route path="/breathing" element={<BreathingPage />} />
            <Route path="/experiments" element={<ExperimentsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
