import { StatusBadge } from "./components/StatusBadge";
import { MetricCard } from "./components/MetricCard";
import { SectionHeader } from "./components/SectionHeader";
import { useAdapterData } from "./hooks/useAdapterData";

export function App() {
  const health = useAdapterData<{
    overall: string;
    alerts: { severity: string; source: string; message: string }[];
    adapter_statuses: { service_name: string; healthy: boolean }[];
  }>("/api/health/");

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "2rem" }}>
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "3rem",
          borderBottom: "1px solid #1f2937",
          paddingBottom: "1.5rem",
        }}
      >
        <div>
          <h1
            style={{
              fontSize: "1.75rem",
              fontWeight: 700,
              letterSpacing: "-0.02em",
            }}
          >
            MinIVess Dashboard
          </h1>
          <p style={{ color: "#9ca3af", fontSize: "0.875rem", marginTop: 4 }}>
            Biomedical Segmentation MLOps Platform
          </p>
        </div>
        <StatusBadge status={health.data?.overall ?? "loading"} />
      </header>

      <SectionHeader title="Platform Health" />
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
          gap: "1rem",
          marginBottom: "2rem",
        }}
      >
        {health.data?.adapter_statuses?.map((s) => (
          <MetricCard
            key={s.service_name}
            label={s.service_name}
            value={s.healthy ? "Online" : "Offline"}
            color={s.healthy ? "#14b8a6" : "#ef4444"}
          />
        )) ?? (
          <MetricCard label="Loading..." value="..." color="#9ca3af" />
        )}
      </div>

      {health.data?.alerts && health.data.alerts.length > 0 && (
        <>
          <SectionHeader title="Alerts" />
          <div style={{ marginBottom: "2rem" }}>
            {health.data.alerts.map((a, i) => (
              <div
                key={i}
                style={{
                  background: "#1c1917",
                  border: "1px solid #7c2d12",
                  borderRadius: 8,
                  padding: "0.75rem 1rem",
                  marginBottom: "0.5rem",
                  fontSize: "0.875rem",
                }}
              >
                <strong style={{ color: "#ef4444" }}>{a.source}</strong>:{" "}
                {a.message}
              </div>
            ))}
          </div>
        </>
      )}

      <SectionHeader title="Experiments" />
      <p style={{ color: "#9ca3af", fontSize: "0.875rem" }}>
        Connect to MLflow, Prefect, and BentoML services to see live data.
        Start the Docker stack and refresh.
      </p>
    </div>
  );
}
