interface MetricCardProps {
  label: string;
  value: string;
  color?: string;
}

export function MetricCard({ label, value, color = "#e5e7eb" }: MetricCardProps) {
  return (
    <div
      style={{
        background: "#111827",
        border: "1px solid #1f2937",
        borderRadius: 8,
        padding: "1.25rem",
      }}
    >
      <div style={{ fontSize: "0.75rem", color: "#9ca3af", marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ fontSize: "1.25rem", fontWeight: 700, color }}>{value}</div>
    </div>
  );
}
