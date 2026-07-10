import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import { DashboardPage } from './pages/DashboardPage'
import { SessionsPage } from './pages/SessionsPage'
import { SessionPage } from './pages/SessionPage'
import { DspInspectorPage } from './pages/DspInspectorPage'
import { VitalsPage } from './pages/VitalsPage'
import { SleepPage } from './pages/SleepPage'
import { ExperimentsPage } from './pages/ExperimentsPage'
import './App.css'

export default function App() {
  return (
    <BrowserRouter>
      <div className="layout">
        <nav className="sidebar">
          <div className="brand">csi-sensing</div>
          <NavLink to="/" end>Dashboard</NavLink>
          <NavLink to="/sessions">Sessions</NavLink>
          <NavLink to="/vitals">Vitals</NavLink>
          <NavLink to="/sleep">Sleep</NavLink>
          <NavLink to="/dsp">DSP inspector</NavLink>
          <NavLink to="/experiments">Experiments</NavLink>
        </nav>
        <main className="content">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/sessions" element={<SessionsPage />} />
            <Route path="/sessions/:id" element={<SessionPage />} />
            <Route path="/vitals" element={<VitalsPage />} />
            <Route path="/sleep" element={<SleepPage />} />
            <Route path="/dsp" element={<DspInspectorPage />} />
            <Route path="/experiments" element={<ExperimentsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
