/**
 * EXODUS Galaxy Explorer — Candidate Highlight Panel
 *
 * Shows prime candidate targets with their full research stories,
 * peer review results, and follow-up recommendations.
 * Appears as a collapsible panel between the convergence panel and detail panel.
 */

import { useState } from 'react';
import type { Candidate, ScoredTarget } from '../types';

interface CandidatePanelProps {
  candidates: Candidate[];
  onGotoCandidate: (target: ScoredTarget) => void;
}

const CHANNEL_COLORS: Record<string, string> = {
  ir_excess: '#ff9800',
  proper_motion_anomaly: '#e040fb',
  radio_emission: '#00e5ff',
  uv_anomaly: '#7c4dff',
  ir_variability: '#ff5722',
};

const CHANNEL_LABELS: Record<string, string> = {
  ir_excess: 'IR Excess',
  proper_motion_anomaly: 'PM Anomaly',
  radio_emission: 'Radio',
  uv_anomaly: 'UV Anomaly',
  ir_variability: 'IR Variability',
};

function PeerReviewBar({ confirmed, total, confidence }: { confirmed: number; total: number; confidence: number }) {
  const pct = Math.round((confirmed / total) * 100);
  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
        <span style={{ fontSize: 10, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
          Peer Review
        </span>
        <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
          {confirmed}/{total} confirmed ({confidence}%)
        </span>
      </div>
      <div style={{ width: '100%', height: 4, background: 'rgba(255,255,255,0.06)', borderRadius: 2, overflow: 'hidden' }}>
        <div
          style={{
            width: `${pct}%`,
            height: '100%',
            background: confidence >= 75 ? '#44cc44' : confidence >= 50 ? '#ffaa00' : '#ff4444',
            borderRadius: 2,
            transition: 'width 0.3s ease',
          }}
        />
      </div>
    </div>
  );
}

function CandidateCard({
  candidate,
  isExpanded,
  onToggle,
  onGoto,
}: {
  candidate: Candidate;
  isExpanded: boolean;
  onToggle: () => void;
  onGoto: () => void;
}) {
  return (
    <div
      style={{
        background: 'rgba(255,215,0,0.04)',
        border: '1px solid rgba(255,215,0,0.15)',
        borderRadius: 8,
        marginBottom: 8,
        overflow: 'hidden',
      }}
    >
      {/* Card header — always visible */}
      <button
        onClick={onToggle}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          width: '100%',
          padding: '10px 12px',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          textAlign: 'left',
        }}
      >
        {/* Rank badge */}
        <div
          style={{
            width: 28,
            height: 28,
            borderRadius: '50%',
            background: candidate.verdict === 'ELIMINATED'
              ? 'linear-gradient(135deg, #ff4444, #cc2222)'
              : candidate.rank === 1
                ? 'linear-gradient(135deg, #ffd700, #ff9800)'
                : 'linear-gradient(135deg, #c0c0c0, #888)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
            fontWeight: 700,
            fontSize: 13,
            color: '#000',
          }}
        >
          {candidate.rank}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-bright)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {candidate.name}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', marginTop: 1 }}>
            {candidate.spectral_type} · {candidate.distance_pc} pc · Score {candidate.exodus_score}
          </div>
        </div>
        {/* Channel badges */}
        <div style={{ display: 'flex', gap: 3, flexShrink: 0 }}>
          {candidate.active_channels.map((ch) => (
            <span
              key={ch}
              style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                background: CHANNEL_COLORS[ch] || '#888',
              }}
              title={CHANNEL_LABELS[ch] || ch}
            />
          ))}
        </div>
        <span style={{ color: 'var(--text-dim)', fontSize: 12, flexShrink: 0 }}>
          {isExpanded ? '▾' : '▸'}
        </span>
      </button>

      {/* Expanded content */}
      {isExpanded && (
        <div style={{ padding: '0 12px 12px' }}>
          {/* Headline */}
          <div
            style={{
              fontSize: 11,
              color: '#ffd700',
              marginBottom: 10,
              lineHeight: 1.4,
              fontStyle: 'italic',
            }}
          >
            {candidate.headline}
          </div>

          {/* Channel badges row */}
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 10 }}>
            {candidate.active_channels.map((ch) => (
              <span
                key={ch}
                style={{
                  padding: '2px 8px',
                  borderRadius: 4,
                  fontSize: 9,
                  fontWeight: 600,
                  background: (CHANNEL_COLORS[ch] || '#888') + '22',
                  color: CHANNEL_COLORS[ch] || '#888',
                  border: `1px solid ${(CHANNEL_COLORS[ch] || '#888')}44`,
                }}
              >
                {CHANNEL_LABELS[ch] || ch}
              </span>
            ))}
            <span
              style={{
                padding: '2px 8px',
                borderRadius: 4,
                fontSize: 9,
                fontWeight: 600,
                background: candidate.verdict === 'ELIMINATED' ? '#ff444422' : '#44cc4422',
                color: candidate.verdict === 'ELIMINATED' ? '#ff4444' : '#44cc44',
                border: `1px solid ${candidate.verdict === 'ELIMINATED' ? '#ff444444' : '#44cc4444'}`,
              }}
            >
              {candidate.verdict}
            </span>
          </div>

          {/* Key findings */}
          <div style={{ marginBottom: 10 }}>
            <div style={{ fontSize: 9, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>
              Key Evidence
            </div>
            {candidate.highlights.map((h, i) => (
              <div
                key={i}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 6,
                  fontSize: 10,
                  color: 'var(--text-secondary)',
                  marginBottom: 2,
                  lineHeight: 1.4,
                }}
              >
                <span style={{ color: '#44cc44', flexShrink: 0, fontSize: 8, marginTop: 2 }}>✓</span>
                <span>{h}</span>
              </div>
            ))}
          </div>

          {/* Challenges */}
          {candidate.challenges.length > 0 && (
            <div style={{ marginBottom: 10 }}>
              <div style={{ fontSize: 9, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>
                Challenges
              </div>
              {candidate.challenges.map((c, i) => (
                <div
                  key={i}
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 6,
                    fontSize: 10,
                    color: 'var(--text-dim)',
                    marginBottom: 2,
                    lineHeight: 1.4,
                  }}
                >
                  <span style={{ color: '#ffaa00', flexShrink: 0, fontSize: 8, marginTop: 2 }}>!</span>
                  <span>{c}</span>
                </div>
              ))}
            </div>
          )}

          {/* Next steps */}
          <div style={{ marginBottom: 8 }}>
            <div style={{ fontSize: 9, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>
              Follow-up Needed
            </div>
            {candidate.next_steps.map((s, i) => (
              <div
                key={i}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 6,
                  fontSize: 10,
                  color: 'var(--text-secondary)',
                  marginBottom: 2,
                  lineHeight: 1.4,
                }}
              >
                <span style={{ color: '#4488ff', flexShrink: 0 }}>{i + 1}.</span>
                <span>{s}</span>
              </div>
            ))}
          </div>

          {/* TESS lightcurve result */}
          {candidate.tess && (
            <div style={{
              marginBottom: 10,
              padding: '6px 8px',
              background: 'rgba(255,255,255,0.02)',
              borderRadius: 4,
              border: '1px solid var(--border)',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <span style={{ fontSize: 9, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  TESS FFI
                </span>
                <span style={{
                  padding: '1px 6px',
                  borderRadius: 3,
                  fontSize: 8,
                  fontWeight: 600,
                  background: candidate.tess.status === 'INCONCLUSIVE' ? 'rgba(255,170,0,0.15)' :
                    candidate.tess.status === 'NO_BINARY_EVIDENCE' ? 'rgba(68,204,68,0.15)' : 'rgba(255,68,68,0.15)',
                  color: candidate.tess.status === 'INCONCLUSIVE' ? '#ffaa00' :
                    candidate.tess.status === 'NO_BINARY_EVIDENCE' ? '#44cc44' : '#ff4444',
                  border: `1px solid ${candidate.tess.status === 'INCONCLUSIVE' ? 'rgba(255,170,0,0.3)' :
                    candidate.tess.status === 'NO_BINARY_EVIDENCE' ? 'rgba(68,204,68,0.3)' : 'rgba(255,68,68,0.3)'}`,
                }}>
                  {candidate.tess.status}
                </span>
              </div>
              <div style={{ fontSize: 9, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', lineHeight: 1.5 }}>
                <div>{candidate.tess.n_sectors} sectors | {candidate.tess.baseline_days}d baseline</div>
                <div>RMS: {candidate.tess.rms_pct}% ({(candidate.tess.rms_ppm / 1000).toFixed(0)}k ppm)</div>
              </div>
              <div style={{ fontSize: 9, color: 'var(--text-dim)', marginTop: 3, lineHeight: 1.3, fontStyle: 'italic' }}>
                {candidate.tess.verdict}
              </div>
            </div>
          )}

          {/* Binary dominance context */}
          {candidate.binary_dominance_context && (
            <div style={{
              marginBottom: 10,
              padding: '6px 8px',
              background: 'rgba(68,136,255,0.06)',
              borderRadius: 4,
              border: '1px solid rgba(68,136,255,0.2)',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 3 }}>
                <span style={{ fontSize: 9, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Binary Dominance
                </span>
                <span style={{
                  padding: '1px 5px',
                  borderRadius: 3,
                  fontSize: 7,
                  fontWeight: 700,
                  background: 'rgba(68,204,68,0.15)',
                  color: '#44cc44',
                  border: '1px solid rgba(68,204,68,0.3)',
                }}>
                  UNIQUE
                </span>
              </div>
              <div style={{ fontSize: 9, color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                {candidate.binary_dominance_context}
              </div>
            </div>
          )}

          {/* Peer review bar */}
          <PeerReviewBar
            confirmed={candidate.peer_review_confirmed}
            total={candidate.peer_review_total}
            confidence={candidate.peer_review_confidence}
          />

          {/* Source + Go to button */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 10 }}>
            <span style={{ fontSize: 9, color: 'var(--text-dim)', fontStyle: 'italic' }}>
              {candidate.source_paper}
            </span>
            <button
              onClick={(e) => { e.stopPropagation(); onGoto(); }}
              style={{
                padding: '4px 12px',
                background: 'rgba(255,215,0,0.15)',
                border: '1px solid rgba(255,215,0,0.3)',
                borderRadius: 4,
                color: '#ffd700',
                fontSize: 10,
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              Go to target →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default function CandidatePanel({ candidates, onGotoCandidate }: CandidatePanelProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(
    candidates.length > 0 ? candidates[0].id : null
  );

  if (candidates.length === 0) return null;

  return (
    <div
      className="glass-panel"
      style={{
        position: 'absolute',
        top: 'calc(var(--topbar-height, 44px) + 10px)',
        right: 12,
        width: 340,
        maxHeight: 'calc(100vh - var(--topbar-height, 44px) - var(--statusbar-height, 32px) - 80px)',
        overflow: 'auto',
        zIndex: 98,
        padding: isCollapsed ? '10px 14px' : '14px',
      }}
    >
      {/* Header */}
      <button
        onClick={() => setIsCollapsed(!isCollapsed)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          width: '100%',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          marginBottom: isCollapsed ? 0 : 10,
          padding: 0,
        }}
      >
        <span style={{ fontSize: 16 }}>🎯</span>
        <span
          style={{
            fontSize: 13,
            fontWeight: 600,
            color: '#ffd700',
            letterSpacing: '0.5px',
            flex: 1,
            textAlign: 'left',
          }}
        >
          Prime Candidates
        </span>
        <span
          style={{
            padding: '2px 8px',
            borderRadius: 10,
            background: 'rgba(255,215,0,0.15)',
            color: '#ffd700',
            fontSize: 11,
            fontWeight: 600,
          }}
        >
          {candidates.length}
        </span>
        <span style={{ color: 'var(--text-dim)', fontSize: 12 }}>
          {isCollapsed ? '▸' : '▾'}
        </span>
      </button>

      {/* Research context blurb */}
      {!isCollapsed && (
        <>
          <div
            style={{
              fontSize: 10,
              color: 'var(--text-dim)',
              lineHeight: 1.5,
              marginBottom: 12,
              padding: '6px 8px',
              background: 'rgba(255,255,255,0.02)',
              borderRadius: 4,
              borderLeft: '2px solid rgba(255,215,0,0.3)',
            }}
          >
            Progressive multi-channel calibration reduces non-binary anomalies to candidates
            requiring follow-up. Candidates are loaded dynamically from pipeline reports.
          </div>

          {/* Candidate cards */}
          {candidates.map((c) => (
            <CandidateCard
              key={c.id}
              candidate={c}
              isExpanded={expandedId === c.id}
              onToggle={() => setExpandedId(expandedId === c.id ? null : c.id)}
              onGoto={() =>
                onGotoCandidate({
                  id: c.id,
                  ra: c.ra,
                  dec: c.dec,
                  distance_pc: c.distance_pc,
                  total_score: c.exodus_score,
                  n_active_channels: c.n_channels,
                  status: 'scored',
                } as ScoredTarget)
              }
            />
          ))}
        </>
      )}
    </div>
  );
}
