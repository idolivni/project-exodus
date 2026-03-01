/**
 * EXODUS Galaxy Explorer — Channel Radar Chart
 *
 * Phase 6: SVG radar/spider chart showing channel activation pattern.
 * Each axis represents a detection channel; the radius shows the score.
 * Provides instant visual fingerprint of a target's anomaly profile.
 */

import type { ChannelDetail } from '../types';

interface ChannelRadarProps {
  channels: Record<string, ChannelDetail>;
  size?: number;
  showLabels?: boolean;
}

const CHANNEL_INFO: { key: string; label: string; shortLabel: string; color: string }[] = [
  { key: 'ir_excess', label: 'IR Excess', shortLabel: 'IR', color: '#ff9800' },
  { key: 'proper_motion_anomaly', label: 'PM Anomaly', shortLabel: 'PM', color: '#e040fb' },
  { key: 'uv_anomaly', label: 'UV Anomaly', shortLabel: 'UV', color: '#7c4dff' },
  { key: 'hr_anomaly', label: 'HR Anomaly', shortLabel: 'HR', color: '#e91e63' },
  { key: 'radio_anomaly', label: 'Radio', shortLabel: 'Rad', color: '#00e5ff' },
  { key: 'radio_emission', label: 'Radio', shortLabel: 'Rad', color: '#00e5ff' },
  { key: 'gaia_photometric_anomaly', label: 'Gaia Phot', shortLabel: 'GP', color: '#00bcd4' },
  { key: 'gaia_photometric', label: 'Gaia Phot', shortLabel: 'GP', color: '#00bcd4' },
  { key: 'transit_anomaly', label: 'Transit', shortLabel: 'Tr', color: '#4caf50' },
  { key: 'habitable_zone_planet', label: 'HZ Prior', shortLabel: 'HZ', color: '#2196f3' },
  { key: 'abundance_anomaly', label: 'Abundance', shortLabel: 'Ab', color: '#8bc34a' },
];

export default function ChannelRadar({ channels, size = 120, showLabels = true }: ChannelRadarProps) {
  // Filter to only channels present in the data
  const activeChannels = CHANNEL_INFO.filter(c => c.key in channels);

  // Deduplicate by shortLabel (radio_anomaly and radio_emission share "Rad")
  const seen = new Set<string>();
  const uniqueChannels = activeChannels.filter(c => {
    if (seen.has(c.shortLabel)) return false;
    seen.add(c.shortLabel);
    return true;
  });

  if (uniqueChannels.length < 3) {
    // Need at least 3 axes for a meaningful radar
    return (
      <div style={{ width: size, height: size / 2, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ display: 'flex', gap: 4 }}>
          {activeChannels.map(c => {
            const ch = channels[c.key];
            return (
              <div
                key={c.key}
                style={{
                  width: 24,
                  height: 24,
                  borderRadius: '50%',
                  background: ch?.active ? `${c.color}33` : 'rgba(255,255,255,0.03)',
                  border: `2px solid ${ch?.active ? c.color : 'rgba(255,255,255,0.1)'}`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: 7,
                  fontWeight: 700,
                  color: ch?.active ? c.color : 'var(--text-dim)',
                }}
                title={`${c.label}: ${ch?.score?.toFixed(3) ?? '—'}`}
              >
                {c.shortLabel}
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  const cx = size / 2;
  const cy = size / 2;
  const radius = (size / 2) - (showLabels ? 20 : 8);
  const n = uniqueChannels.length;

  // Compute polygon points
  const getPoint = (index: number, value: number): [number, number] => {
    const angle = (Math.PI * 2 * index) / n - Math.PI / 2;
    const r = radius * Math.min(1, Math.max(0, value));
    return [cx + r * Math.cos(angle), cy + r * Math.sin(angle)];
  };

  // Background grid rings
  const gridRings = [0.25, 0.5, 0.75, 1.0];

  // Data polygon
  const scores = uniqueChannels.map(c => {
    const ch = channels[c.key];
    return ch ? Math.min(1, ch.score / 1.0) : 0; // Normalize to 0-1
  });

  const dataPoints = scores.map((s, i) => getPoint(i, s));
  const dataPath = dataPoints.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ') + ' Z';

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {/* Background grid */}
      {gridRings.map(r => {
        const ringPoints = Array.from({ length: n }, (_, i) => getPoint(i, r));
        const ringPath = ringPoints.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ') + ' Z';
        return (
          <path
            key={r}
            d={ringPath}
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth={0.5}
          />
        );
      })}

      {/* Axis lines */}
      {uniqueChannels.map((_, i) => {
        const [x, y] = getPoint(i, 1.0);
        return (
          <line
            key={i}
            x1={cx}
            y1={cy}
            x2={x}
            y2={y}
            stroke="rgba(255,255,255,0.08)"
            strokeWidth={0.5}
          />
        );
      })}

      {/* Data polygon (filled) */}
      <path
        d={dataPath}
        fill="rgba(255,200,50,0.15)"
        stroke="rgba(255,200,50,0.7)"
        strokeWidth={1.5}
        strokeLinejoin="round"
      />

      {/* Data points */}
      {dataPoints.map(([x, y], i) => {
        const ch = channels[uniqueChannels[i].key];
        const isActive = ch?.active;
        const color = isActive ? uniqueChannels[i].color : 'rgba(255,255,255,0.2)';
        return (
          <circle
            key={i}
            cx={x}
            cy={y}
            r={isActive ? 3 : 2}
            fill={isActive ? color : 'transparent'}
            stroke={color}
            strokeWidth={isActive ? 1.5 : 0.5}
          />
        );
      })}

      {/* Labels */}
      {showLabels && uniqueChannels.map((c, i) => {
        const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
        const labelR = radius + 12;
        const lx = cx + labelR * Math.cos(angle);
        const ly = cy + labelR * Math.sin(angle);
        const ch = channels[c.key];
        const isActive = ch?.active;
        return (
          <text
            key={i}
            x={lx}
            y={ly}
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize={8}
            fontFamily="var(--font-mono)"
            fontWeight={isActive ? 700 : 400}
            fill={isActive ? c.color : 'rgba(255,255,255,0.3)'}
          >
            {c.shortLabel}
          </text>
        );
      })}
    </svg>
  );
}
