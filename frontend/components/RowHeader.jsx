// The column labels, shown once per section instead of once per row.
export default function RowHeader({ labels }) {
  return (
    <div className="row header-row">
      {labels.map((label) => (
        <div className="col" key={label}>
          <div className="label">{label}</div>
        </div>
      ))}
    </div>
  );
}
