import { Navigate, Outlet } from "react-router-dom";

import { useAuth } from "../auth/AuthContext";

export function RequireAuth() {
  const { token, loading } = useAuth();
  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-hull font-mono text-sm text-mute">
        Checking session…
      </div>
    );
  }
  if (!token) {
    return <Navigate to="/login" replace />;
  }
  return <Outlet />;
}
