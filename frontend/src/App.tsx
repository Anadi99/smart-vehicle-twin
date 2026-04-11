import { useState } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'

import { AppShell } from './components/layout/AppShell'
import { RealtimeProvider } from './context/RealtimeContext'
import { getToken } from './lib/auth'
import DiagnosticsPage from './pages/DiagnosticsPage'
import LapsPage from './pages/LapsPage'
import LoginPage from './pages/LoginPage'
import OverviewPage from './pages/OverviewPage'
import PredictionsPage from './pages/PredictionsPage'
import TelemetryPage from './pages/TelemetryPage'
import Twin3DPage from './pages/Twin3DPage'

const queryClient = new QueryClient()

export default function App() {
  const [authed, setAuthed] = useState(() => !!getToken())

  if (!authed) {
    return <LoginPage onLogin={() => setAuthed(true)} />
  }

  return (
    <QueryClientProvider client={queryClient}>
      <RealtimeProvider>
        <BrowserRouter>
          <AppShell onLogout={() => setAuthed(false)}>
            <Routes>
              <Route path="/" element={<Navigate to="/overview" replace />} />
              <Route path="/overview"    element={<OverviewPage />} />
              <Route path="/telemetry"   element={<TelemetryPage />} />
              <Route path="/predictions" element={<PredictionsPage />} />
              <Route path="/twin"        element={<Twin3DPage />} />
              <Route path="/laps"        element={<LapsPage />} />
              <Route path="/diagnostics" element={<DiagnosticsPage />} />
            </Routes>
          </AppShell>
        </BrowserRouter>
      </RealtimeProvider>
    </QueryClientProvider>
  )
}
