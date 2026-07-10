import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import { DashboardPage } from './pages/DashboardPage'
import { SessionsPage } from './pages/SessionsPage'
import { SessionPage } from './pages/SessionPage'
import { DspInspectorPage } from './pages/DspInspectorPage'
import { VitalsPage } from './pages/VitalsPage'
import { SleepPage } from './pages/SleepPage'
import { ExperimentsPage } from './pages/ExperimentsPage'
import {
  FlaskIcon,
  HeartIcon,
  LayersIcon,
  MoonIcon,
  PulseIcon,
  SlidersIcon,
} from './components/icons'
import './App.css'

export default function App() {
  return (
    <BrowserRouter>
      <div className="layout">
        <nav className="sidebar">
          <div className="brand">
            <svg viewBox="0 0 32 32" className="brand-mark" aria-hidden>
              <rect width="32" height="32" rx="7" fill="#0f1b33" />
              <path
                d="M4 20 L9 20 L12 10 L16 26 L20 14 L23 20 L28 20"
                fill="none"
                stroke="#38bdf8"
                strokeWidth="2.4"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <div>
              <div className="brand-name">csi sensing</div>
              <div className="brand-sub">ambient home monitor</div>
            </div>
          </div>
          <NavLink to="/" end><PulseIcon /> Dashboard</NavLink>
          <NavLink to="/sessions"><LayersIcon /> Sessions</NavLink>
          <NavLink to="/vitals"><HeartIcon /> Vitals</NavLink>
          <NavLink to="/sleep"><MoonIcon /> Sleep</NavLink>
          <NavLink to="/dsp"><SlidersIcon /> DSP inspector</NavLink>
          <NavLink to="/experiments"><FlaskIcon /> Experiments</NavLink>
          <div className="sidebar-footer">alpha - not a medical device</div>
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
