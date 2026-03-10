interface SectionHeaderProps {
  title: string;
}

export function SectionHeader({ title }: SectionHeaderProps) {
  return (
    <h2
      style={{
        fontSize: "0.7rem",
        fontWeight: 700,
        textTransform: "uppercase",
        letterSpacing: "0.1em",
        color: "#14b8a6",
        marginBottom: "1rem",
        borderBottom: "1px solid #1f2937",
        paddingBottom: "0.5rem",
      }}
    >
      {title}
    </h2>
  );
}
