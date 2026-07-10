/** 7x7 confusion matrix as a colored HTML grid. Rows = true, columns = predicted. */

interface Props {
  confusion: number[][]
  classes: string[]
}

export function ConfusionMatrix({ confusion, classes }: Props) {
  const rowTotals = confusion.map((row) => row.reduce((a, b) => a + b, 0) || 1)
  return (
    <div className="confusion-wrap">
      <table className="confusion">
        <thead>
          <tr>
            <th className="corner">true \ pred</th>
            {classes.map((c) => (
              <th key={c} className="pred-head">{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {confusion.map((row, i) => (
            <tr key={classes[i]}>
              <th className="true-head">{classes[i]}</th>
              {row.map((v, j) => {
                const frac = v / rowTotals[i]
                return (
                  <td
                    key={j}
                    style={{
                      background: `rgba(56, 189, 248, ${frac.toFixed(3)})`,
                      color: frac > 0.5 ? '#04121f' : 'var(--muted)',
                    }}
                    title={`${classes[i]} predicted as ${classes[j]}: ${v}`}
                  >
                    {v}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
