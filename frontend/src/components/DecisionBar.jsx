import { useState } from 'react'

export default function DecisionBar({ study, onDecide }) {
  const [reason, setReason] = useState(study.reason || '')

  function decide(decision) {
    onDecide(decision, reason.trim())
  }

  const r = study.result

  return (
    <div className="decision-bar">
      <div className="prediction-summary">
        <strong>{r.prediction}</strong> · probability {(r.probability * 100).toFixed(1)}% ·
        threshold {r.threshold_used} · {r.inference_ms} ms · model {r.model_version}
      </div>
      <div className="decision-controls">
        <button className="btn-approve" onClick={() => decide('approved')}>
          Approve
        </button>
        <button className="btn-reject" onClick={() => decide('rejected')}>
          Reject
        </button>
        <input
          type="text"
          placeholder="reason (free text)"
          value={reason}
          onChange={(e) => setReason(e.target.value)}
        />
        {study.decision && (
          <span className={`badge ${study.decision === 'approved' ? 'badge-ok' : 'badge-bad'}`}>
            {study.decision}
          </span>
        )}
      </div>
      <p className="hint">
        Decisions are kept in this session only until the audit endpoint is wired to audit.db.
      </p>
    </div>
  )
}
