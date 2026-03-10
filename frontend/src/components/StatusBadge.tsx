interface StatusBadgeProps {
  status: string;
}

const STATUS_COLORS: Record<string, string> = {
  healthy: "#14b8a6",
  degraded: "#f59e0b",
  unhealthy: "#ef4444",
  loading: "#6b7280",
};

export function StatusBadge({ status }: StatusBadgeProps) {
  const color = STATUS_COLORS[status] ?? "#6b7280";
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: "4px 12px",
        borderRadius: 999,
        border: `1px solid ${color}`,
        fontSize: "0.75rem",
        fontWeight: 600,
        textTransform: "uppercase",
        letterSpacing: "0.05em",
        color,
      }}
    >
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          background: color,
        }}
      />
      {status}
    </span>
  );
}
