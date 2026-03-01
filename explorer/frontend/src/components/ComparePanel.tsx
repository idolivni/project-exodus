/**
 * EXODUS Galaxy Explorer — Compare Panel
 *
 * Phase 6: Side-by-side target comparison.
 * Shows two targets' channel fingerprints and key metrics in parallel
 * for rapid visual comparison.
 *
 * Features:
 * - Dual radar charts (overlaid or side-by-side)
 * - Key metrics comparison (score, channels, distance, risk)
 * - Channel-by-channel difference highlighting
 * - Add/swap targets from selection
 */

import { useMemo } from 'react';
import type { ScoredTarget, ChannelDetail } from '../types';
import ChannelRadar from './ChannelRadar';

interface ComparePanelProps {
  targetA: ScoredTarget | null;
  targetB: ScoredTarget | null;
  onClose: () => void;
  onClearSlot: (slot: 'A' | 'B') => void;
}

const CHANNEL_LABELS: Record<string, { label: string; color: string }> = {
  ir_excess: { label: 'IR Excess', color: '#ff9800' },
  ir_variability: { label: 'IR Var', color: '#ff6b35' },
  proper_motion_anomaly: { label: 'PM Anomaly', color: '#e040fb' },
  pm_anomaly: { label: 'PM Anomaly', color: '#e040fb' },
  gaia_photometric: { label: 'Gaia Phot', color: '#00bcd4' },
  gaia_photometric_anomaly: { label: 'Gaia Phot', color: '#00bcd4' },
  transit_anomaly: { label: 'Transit', color: '#4caf50' },
  radio_anomaly: { label: 'Radio', color: '#00e5ff' },
  radio_emission: { label: 'Radio', color: '#00e5ff' },
  uv_anomaly: { label: 'UV Anomaly', color: '#7c4dff' },
  hr_anomaly: { label: 'HR Anomaly', color: '#e91e63' },
  abundance_anomaly: { label: 'Abundance', color: '#8bc34a' },
  habitable_zone_planet: { label: 'HZ Prior', color: '#2196f3' },
  hz_prior: { label: 'HZ Prior', color: '#2196f3' },
};

/** Get all channels from both targets (union) */
function getAllChannelKeys(a: ScoredTarget | null, b: ScoredTarget | null): string[] {
  const keys = new Set<string>();
  if (a?.channel_details) Object.keys(a.channel_details).forEach(k => keys.add(k));
  if (b?.channel_details) Object.keys(b.channel_details).forEach(k => keys.add(k));
  // Sort by label for consistent display
  return Array.from(keys).sort((x, y) => {
    const lx = CHANNEL_LABELS[x]?.label ?? x;
    const ly = CHANNEL_LABELS[y]?.label ?? y;
    return lx.localeCompare(ly);
  });
}

function MetricRow({ label, valueA, valueB, highlight }: {
  label: string;
  valueA: string;
  valueB: string;
  highlight?: 'A' | 'B' | null;
}) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 4,
        padding: '3px 0',
        fontSize: 10,
        borderBottom: '1px solid rgba(255,255,255,0.04)',
      }}
    >
      <span
        className="mono"
        style={{
          flex: 1,
          textAlign: 'right',
          color: highlight === 'A' ? '#ffd700' : 'var(--text-secondary)',
          fontWeight: highlight === 'A' ? 600 : 400,
        }}
      >
        {valueA}
      </span>
      <span
        style={{
          width: 70,
          textAlign: 'center',
          color: 'var(--text-dim)',
          fontSize: 9,
          textTransform: 'uppercase',
          letterSpacing: '0.3px',
          flexShrink: 0,
        }}
      >
        {label}
      </span>
      <span
        className="mono"
        style={{
          flex: 1,
          textAlign: 'left',
          color: highlight === 'B' ? '#ffd700' : 'var(--text-secondary)',
          fontWeight: highlight === 'B' ? 600 : 400,
        }}
      >
        {valueB}
      </span>
    </div>
  );
}

function ChannelCompareRow({ channelKey, chA, chB }: {
  channelKey: string;
  chA?: ChannelDetail;
  chB?: ChannelDetail;
}) {
  const info = CHANNEL_LABELS[channelKey] || { label: channelKey, color: '#888' };
  const scoreA = chA?.score ?? 0;
  const scoreB = chB?.score ?? 0;
  const maxScore = Math.max(scoreA, scoreB, 0.01);

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4, padding: '2px 0' }}>
      {/* Bar A (right-aligned) */}
      <div style={{ flex: 1, display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: 4 }}>
        <span
          className="mono"
          style={{
            fontSize: 9,
            color: chA?.active ? 'var(--text-secondary)' : 'var(--text-dim)',
          }}
        >
          {scoreA > 0 ? scoreA.toFixed(2) : '—'}
        </span>
        <div
          style={{
            width: `${Math.max(2, (scoreA / maxScore) * 50)}px`,
            height: 6,
            borderRadius: 2,
            background: chA?.active ? info.color : 'rgba(255,255,255,0.06)',
            opacity: chA?.active ? 0.8 : 0.3,
            transition: 'width 0.2s ease',
          }}
        />
      </div>

      {/* Channel label */}
      <span
        style={{
          width: 52,
          textAlign: 'center',
          fontSize: 8,
          fontWeight: 600,
          color: (chA?.active || chB?.active) ? info.color : 'var(--text-dim)',
          letterSpacing: '0.3px',
          flexShrink: 0,
        }}
      >
        {info.label}
      </span>

      {/* Bar B (left-aligned) */}
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 4 }}>
        <div
          style={{
            width: `${Math.max(2, (scoreB / maxScore) * 50)}px`,
            height: 6,
            borderRadius: 2,
            background: chB?.active ? info.color : 'rgba(255,255,255,0.06)',
            opacity: chB?.active ? 0.8 : 0.3,
            transition: 'width 0.2s ease',
          }}
        />
        <span
          className="mono"
          style={{
            fontSize: 9,
            color: chB?.active ? 'var(--text-secondary)' : 'var(--text-dim)',
          }}
        >
          {scoreB > 0 ? scoreB.toFixed(2) : '—'}
        </span>
      </div>
    </div>
  );
}

function SlotHeader({ target, label, onClear }: {
  target: ScoredTarget | null;
  label: string;
  onClear: () => void;
}) {
  if (!target) {
    return (
      <div style={{ flex: 1, textAlign: 'center' }}>
        <div style={{ fontSize: 10, color: 'var(--text-dim)', fontStyle: 'italic' }}>
          Click a target to fill Slot {label}
        </div>
      </div>
    );
  }

  return (
    <div style={{ flex: 1, textAlign: 'center', position: 'relative' }}>
      <div
        className="mono"
        style={{
          fontSize: 10,
          fontWeight: 600,
          color: 'var(--text-bright)',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          padding: '0 14px',
        }}
        title={target.id}
      >
        {target.id.length > 18 ? target.id.slice(-16) : target.id}
      </div>
      <div
        className="mono"
        style={{ fontSize: 9, color: 'var(--text-dim)', marginTop: 1 }}
      >
        {target.total_score?.toFixed(2)} · {target.n_active_channels}ch
        {target.distance_pc ? ` · ${target.distance_pc.toFixed(0)}pc` : ''}
      </div>
      <button
        onClick={onClear}
        style={{
          position: 'absolute',
          top: -2,
          right: 0,
          background: 'none',
          border: 'none',
          color: 'var(--text-dim)',
          cursor: 'pointer',
          fontSize: 11,
          padding: '0 2px',
          opacity: 0.5,
        }}
        title={`Clear slot ${label}`}
      >
        ×
      </button>
    </div>
  );
}

export default function ComparePanel({ targetA, targetB, onClose, onClearSlot }: ComparePanelProps) {
  const allChannels = useMemo(
    () => getAllChannelKeys(targetA, targetB),
    [targetA, targetB]
  );

  // Compute which target "wins" each metric
  const scoreHighlight = useMemo(() => {
    if (!targetA || !targetB) return null;
    if (targetA.total_score > targetB.total_score) return 'A' as const;
    if (targetB.total_score > targetA.total_score) return 'B' as const;
    return null;
  }, [targetA, targetB]);

  const channelHighlight = useMemo(() => {
    if (!targetA || !targetB) return null;
    if ((targetA.n_active_channels ?? 0) > (targetB.n_active_channels ?? 0)) return 'A' as const;
    if ((targetB.n_active_channels ?? 0) > (targetA.n_active_channels ?? 0)) return 'B' as const;
    return null;
  }, [targetA, targetB]);

  const unexHighlight = useMemo(() => {
    if (!targetA || !targetB) return null;
    const ua = targetA.unexplainability_score ?? 0;
    const ub = targetB.unexplainability_score ?? 0;
    if (ua > ub) return 'A' as const;
    if (ub > ua) return 'B' as const;
    return null;
  }, [targetA, targetB]);

  // If nothing to show
  if (!targetA && !targetB) return null;

  const channelsA = targetA?.channel_details ?? {};
  const channelsB = targetB?.channel_details ?? {};

  return (
    <div
      className="glass-panel"
      style={{
        position: 'absolute',
        bottom: 44,
        left: '50%',
        transform: 'translateX(-50%)',
        width: 380,
        maxHeight: 400,
        overflow: 'auto',
        zIndex: 99,
        padding: '12px',
        animation: 'slideInRight 0.2s ease-out',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 10,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 12 }}>⚖️</span>
          <span
            style={{
              fontSize: 12,
              fontWeight: 600,
              color: 'var(--text-bright)',
              letterSpacing: '0.3px',
            }}
          >
            Compare Targets
          </span>
        </div>
        <button
          onClick={onClose}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--text-dim)',
            cursor: 'pointer',
            fontSize: 16,
            padding: '0 2px',
          }}
        >
          ×
        </button>
      </div>

      {/* Slot headers */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
        <SlotHeader target={targetA} label="A" onClear={() => onClearSlot('A')} />
        <div style={{ width: 1, background: 'var(--border)', flexShrink: 0 }} />
        <SlotHeader target={targetB} label="B" onClear={() => onClearSlot('B')} />
      </div>

      {/* Dual radar charts (side by side) */}
      {targetA && targetB && (
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            gap: 8,
            padding: '4px 0 8px',
            marginBottom: 8,
            borderTop: '1px solid var(--border)',
            borderBottom: '1px solid var(--border)',
          }}
        >
          <div style={{ textAlign: 'center' }}>
            <ChannelRadar channels={channelsA} size={110} showLabels={true} />
          </div>
          <div style={{ textAlign: 'center' }}>
            <ChannelRadar channels={channelsB} size={110} showLabels={true} />
          </div>
        </div>
      )}

      {/* Single radar when only one target */}
      {((targetA && !targetB) || (!targetA && targetB)) && (
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            padding: '4px 0 8px',
            marginBottom: 8,
            borderTop: '1px solid var(--border)',
            borderBottom: '1px solid var(--border)',
          }}
        >
          <ChannelRadar
            channels={(targetA ?? targetB)!.channel_details ?? {}}
            size={130}
            showLabels={true}
          />
        </div>
      )}

      {/* Key metrics comparison */}
      {targetA && targetB && (
        <>
          <div
            style={{
              fontSize: 9,
              color: 'var(--text-dim)',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              marginBottom: 4,
            }}
          >
            Key Metrics
          </div>
          <MetricRow
            label="Score"
            valueA={targetA.total_score?.toFixed(3) ?? '—'}
            valueB={targetB.total_score?.toFixed(3) ?? '—'}
            highlight={scoreHighlight}
          />
          <MetricRow
            label="Channels"
            valueA={String(targetA.n_active_channels ?? 0)}
            valueB={String(targetB.n_active_channels ?? 0)}
            highlight={channelHighlight}
          />
          <MetricRow
            label="Distance"
            valueA={targetA.distance_pc ? `${targetA.distance_pc.toFixed(1)} pc` : '—'}
            valueB={targetB.distance_pc ? `${targetB.distance_pc.toFixed(1)} pc` : '—'}
          />
          <MetricRow
            label="Unex"
            valueA={targetA.unexplainability_score != null ? targetA.unexplainability_score.toFixed(3) : '—'}
            valueB={targetB.unexplainability_score != null ? targetB.unexplainability_score.toFixed(3) : '—'}
            highlight={unexHighlight}
          />
          <MetricRow
            label="Risk"
            valueA={targetA.red_team_risk ?? '—'}
            valueB={targetB.red_team_risk ?? '—'}
          />
          <MetricRow
            label="FDR"
            valueA={targetA.fdr_significant ? 'YES' : 'no'}
            valueB={targetB.fdr_significant ? 'YES' : 'no'}
          />
        </>
      )}

      {/* Channel-by-channel comparison (butterfly chart) */}
      {targetA && targetB && allChannels.length > 0 && (
        <div style={{ marginTop: 10 }}>
          <div
            style={{
              fontSize: 9,
              color: 'var(--text-dim)',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              marginBottom: 6,
            }}
          >
            Channel Comparison
          </div>
          {allChannels.map(key => (
            <ChannelCompareRow
              key={key}
              channelKey={key}
              chA={channelsA[key]}
              chB={channelsB[key]}
            />
          ))}
        </div>
      )}

      {/* Help text when only one slot filled */}
      {((!targetA && targetB) || (targetA && !targetB)) && (
        <div
          style={{
            padding: 12,
            textAlign: 'center',
            fontSize: 10,
            color: 'var(--text-dim)',
            fontStyle: 'italic',
          }}
        >
          Click another target on the sky map to fill the second comparison slot.
        </div>
      )}
    </div>
  );
}
