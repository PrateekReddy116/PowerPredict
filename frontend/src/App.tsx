import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import { AuthProvider, useAuth } from "./auth/AuthContext";
import { AppShell } from "./components/AppShell";
import { RequireAuth } from "./components/RequireAuth";
import { AnomaliesPage } from "./pages/Anomalies";
import { BuildingsPage } from "./pages/Buildings";
import { ExplainPage } from "./pages/Explain";
import { ForecastPage } from "./pages/Forecast";
import { PredictPage } from "./pages/Predict";
import { IotPage } from "./pages/Iot";
import { LandingPage } from "./pages/Landing";
import { LoginPage } from "./pages/Login";
import { ModelsPage } from "./pages/Models";
import { OverviewPage } from "./pages/Overview";

/** Unauthenticated `/` = public landing; signed-in users go to Overview. */
function RootEntry() {
  const { token, loading } = useAuth();
  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-hull font-mono text-sm text-mute">
        Checking session…
      </div>
    );
  }
  if (token) {
    return <Navigate to="/overview" replace />;
  }
  return <LandingPage />;
}

export function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<RootEntry />} />
          <Route path="/login" element={<LoginPage />} />
          <Route element={<RequireAuth />}>
            <Route element={<AppShell />}>
              <Route path="/overview" element={<OverviewPage />} />
              <Route path="/forecast" element={<ForecastPage />} />
              <Route path="/predict" element={<PredictPage />} />
              <Route path="/models" element={<ModelsPage />} />
              <Route path="/explain" element={<ExplainPage />} />
              <Route path="/anomalies" element={<AnomaliesPage />} />
              <Route path="/iot" element={<IotPage />} />
              <Route path="/buildings" element={<BuildingsPage />} />
            </Route>
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
