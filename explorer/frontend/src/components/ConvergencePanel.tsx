/**
 * EXODUS Galaxy Explorer — Convergence Panel (Enhanced)
 *
 * Phase 3+: Controls for the convergence engine + enhanced zone list.
 *
 * Features:
 * - Convergence radius slider (1 arcmin to 5 degrees)
 * - Minimum channels slider (2-6)
 * - Include multi-messenger toggle
 * - Sort controls: by score, signal count, or unexplainability
 * - Filter: all / unexplained only / explained only
 * - Unexplainability score display per zone with color coding
 * - At-a-glance status badges (UNEXPLAINED / EXPLAINED / UNKNOWN)
 * - Click-to-fly navigation
 */

import { useState, useMemo } from 'react';
import type { ConvergenceZone, ConvergenceConfig } from '../utils/convergence';
import { DEFAULT_CONVERGENCE_CONFIG, convergenceColor } from '../utils/convergence';
import type { ScoredTarget } from '../types';

interface ConvergencePanelProps {
  zones: ConvergenceZone[];
  config: ConvergenceConfig;
  onConfigChange: (config: ConvergenceConfig) => void;
  onGotoZone: (target: ScoredTarget) => void;
  computing: boolean;
}

type SortMode = 'signals' | 'score' | 'unexplainability' | 'distance';
type FilterMode = 'all' | 'unexplained' | 'partial' | 'explained';

/** Format angular radius for display */
function formatRadius(deg: number): string {
  if (deg >= 1) return `${deg.toFixed(1)}°`;
  return `${(deg * 60).toFixed(0)}'`;
}

/** Get status badge for a zone */
function getZoneStatus(zone: ConvergenceZone): {
  label: string;
  color: string;
  bgColor: string;
  borderColor: string;
} {
  if (zone.unexplainability == null) {
    return {
      label: 'UNSCORED',
      color: '#888',
      bgColor: 'rgba(136,136,136,0.1)',
      borderColor: 'rgba(136,136,136,0.25)',
    };
  }
  if (zone.unexplainability >= 0.5) {
    return {
      label: 'UNEXPLAINED',
      color: '#ff4444',
      bgColor: 'rgba(255,68,68,0.12)',
      borderColor: 'rgba(255,68,68,0.3)',
    };
  }
  if (zone.unexplainability >= 0.2) {
    return {
      label: 'PARTIAL',
      color: '#ffaa00',
      bgColor: 'rgba(255,170,0,0.1)',
      borderColor: 'rgba(255,170,0,0.25)',
    };
  }
  return {
    label: 'EXPLAINED',
    color: '#44cc44',
    bgColor: 'rgba(68,204,68,0.1)',
    borderColor: 'rgba(68,204,68,0.25)',
  };
}

/** Unexplainability score color */
function unexColor(val: number | undefined): string {
  if (val == null) return '#555';
  if (val >= 0.5) return '#ff4444';
  if (val >= 0.2) return '#ffaa00';
  return '#44cc44';
}

export default function ConvergencePanel({
  zones,
  config,
  onConfigChange,
  onGotoZone,
  computing,
}: ConvergencePanelProps) {
  const [collapsed, setCollapsed] = useState(true);
  const [sortMode, setSortMode] = useState<SortMode>('signals');
  const [filterMode, setFilterMode] = useState<FilterMode>('all');
  const [channelFilter, setChannelFilter] = useState<Set<string>>(new Set());
  const [campaignFilter, setCampaignFilter] = useState<string>('all');

  // Compute summary stats + unique channels + campaigns + per-channel zone counts
  const stats = useMemo(() => {
    const unexplained = zones.filter(z => z.unexplainability != null && z.unexplainability >= 0.5).length;
    const explained = zones.filter(z => z.is_explained).length;
    const partial = zones.filter(z => z.unexplainability != null && z.unexplainability >= 0.2 && z.unexplainability < 0.5).length;
    const fdr = zones.filter(z => z.fdr_significant).length;

    // Unique channel names across all zones + per-channel zone counts
    const channelCounts = new Map<string, number>();
    zones.forEach(z => z.channels.forEach(ch => {
      channelCounts.set(ch, (channelCounts.get(ch) ?? 0) + 1);
    }));

    // Extract campaign prefixes from zone IDs
    const campaignSet = new Set<string>();
    zones.forEach(z => {
      // IDs like "CAMPAIGN_RA123.456..." or "GAIA_520377..."
      const match = z.id.match(/^([A-Z_]+?)_(?:RA|GAIA_?\d|[0-9])/i);
      if (match) {
        campaignSet.add(match[1].toUpperCase());
      }
    });

    return {
      unexplained, explained, partial, fdr, total: zones.length,
      channels: Array.from(channelCounts.keys()).sort(),
      channelCounts,
      campaigns: Array.from(campaignSet).sort(),
    };
  }, [zones]);

  // Filter zones (status + channel + campaign)
  const filteredZones = useMemo(() => {
    let result = zones;

    // Status filter
    if (filterMode === 'unexplained') {
      result = result.filter(z => z.unexplainability != null && z.unexplainability >= 0.5);
    } else if (filterMode === 'partial') {
      result = result.filter(z => z.unexplainability != null && z.unexplainability >= 0.2 && z.unexplainability < 0.5);
    } else if (filterMode === 'explained') {
      result = result.filter(z => z.is_explained);
    }

    // Channel filter (AND logic: zone must have ALL selected channels)
    if (channelFilter.size > 0) {
      result = result.filter(z => {
        const zoneChannels = new Set(z.channels);
        return Array.from(channelFilter).every(ch => zoneChannels.has(ch));
      });
    }

    // Campaign filter
    if (campaignFilter !== 'all') {
      result = result.filter(z => z.id.toUpperCase().startsWith(campaignFilter + '_'));
    }

    return result;
  }, [zones, filterMode, channelFilter, campaignFilter]);

  // Sort zones
  const sortedZones = useMemo(() => {
    const sorted = [...filteredZones];
    switch (sortMode) {
      case 'score':
        sorted.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
        break;
      case 'unexplainability':
        sorted.sort((a, b) => (b.unexplainability ?? -1) - (a.unexplainability ?? -1));
        break;
      case 'distance':
        sorted.sort((a, b) => {
          const dA = a.anchor_target?.distance_pc;
          const dB = b.anchor_target?.distance_pc;
          if (dA == null && dB == null) return 0;
          if (dA == null) return 1;  // nulls last
          if (dB == null) return -1;
          return dA - dB;
        });
        break;
      case 'signals':
      default:
        sorted.sort((a, b) => {
          if (b.n_channels !== a.n_channels) return b.n_channels - a.n_channels;
          return (b.score ?? 0) - (a.score ?? 0);
        });
        break;
    }
    return sorted;
  }, [filteredZones, sortMode]);

  return (
    <div
      className="glass-panel"
      style={{
        position: 'absolute',
        top: 'calc(var(--topbar-height, 44px) + 76px)',
        left: 12,
        width: collapsed ? 'auto' : 300,
        maxHeight: collapsed ? 'auto' : 'calc(100vh - var(--topbar-height, 44px) - var(--statusbar-height, 32px) - 120px)',
        overflow: 'hidden',
        zIndex: 99,
        padding: collapsed ? '6px 10px' : 0,
        transition: 'all 0.2s ease',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: collapsed ? 0 : '10px 12px 8px',
          cursor: 'pointer',
          flexShrink: 0,
        }}
        onClick={() => setCollapsed(!collapsed)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 14 }}>🎯</span>
          <span
            style={{
              fontSize: 13,
              fontWeight: 600,
              color: 'var(--text-bright)',
              letterSpacing: '0.3px',
            }}
          >
            Convergence
          </span>
          {zones.length > 0 && (
            <span
              className="mono"
              style={{
                fontSize: 10,
                padding: '1px 6px',
                borderRadius: 8,
                background: 'rgba(255,170,0,0.2)',
                color: '#ffaa00',
                fontWeight: 600,
              }}
            >
              {zones.length}
            </span>
          )}
          {/* Mini unexplained + partial count badge */}
          {!collapsed && (stats.unexplained > 0 || stats.partial > 0) && (
            <span
              className="mono"
              style={{
                fontSize: 9,
                padding: '1px 5px',
                borderRadius: 8,
                background: stats.unexplained > 0 ? 'rgba(255,68,68,0.15)' : 'rgba(255,170,0,0.15)',
                color: stats.unexplained > 0 ? '#ff4444' : '#ffaa00',
                fontWeight: 600,
              }}
            >
              {stats.unexplained > 0 ? `${stats.unexplained} unex` : ''}{stats.unexplained > 0 && stats.partial > 0 ? ' · ' : ''}{stats.partial > 0 ? `${stats.partial} partial` : ''}
            </span>
          )}
        </div>
        <span style={{ color: 'var(--text-dim)', fontSize: 12 }}>
          {collapsed ? '▸' : '▾'}
        </span>
      </div>

      {!collapsed && (
        <>
          {/* Controls section */}
          <div style={{ padding: '0 12px 10px', flexShrink: 0 }}>
            {/* Radius slider */}
            <div style={{ marginBottom: 8 }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontSize: 11,
                  color: 'var(--text-secondary)',
                  marginBottom: 3,
                }}
              >
                <span>Match Radius</span>
                <span className="mono" style={{ color: 'var(--text-bright)' }}>
                  {formatRadius(config.radius_deg)}
                </span>
              </div>
              <input
                type="range"
                min={1}
                max={300}
                value={config.radius_deg * 60}
                onChange={(e) =>
                  onConfigChange({ ...config, radius_deg: Number(e.target.value) / 60 })
                }
                style={{
                  width: '100%',
                  height: 3,
                  accentColor: '#ffaa00',
                  cursor: 'pointer',
                }}
              />
              {/* Per-source radii info */}
              {config.include_mm && (
                <div
                  style={{
                    marginTop: 3,
                    padding: '2px 6px',
                    borderRadius: 4,
                    background: 'rgba(255,255,255,0.03)',
                    fontSize: 9,
                    color: 'var(--text-dim)',
                    lineHeight: 1.5,
                  }}
                >
                  <span style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>Per-source:</span>{' '}
                  Fermi {formatRadius(0.3 * (config.radius_deg / 0.5))},{' '}
                  IceCube {formatRadius(1.5 * (config.radius_deg / 0.5))},{' '}
                  FRB {formatRadius(0.8 * (config.radius_deg / 0.5))},{' '}
                  Pulsar {formatRadius(0.1 * (config.radius_deg / 0.5))}
                </div>
              )}
            </div>

            {/* Min channels slider */}
            <div style={{ marginBottom: 8 }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontSize: 11,
                  color: 'var(--text-secondary)',
                  marginBottom: 3,
                }}
              >
                <span>Min Channels</span>
                <span className="mono" style={{ color: 'var(--text-bright)' }}>
                  {config.min_channels}
                </span>
              </div>
              <input
                type="range"
                min={2}
                max={6}
                value={config.min_channels}
                onChange={(e) =>
                  onConfigChange({ ...config, min_channels: Number(e.target.value) })
                }
                style={{
                  width: '100%',
                  height: 3,
                  accentColor: '#ffaa00',
                  cursor: 'pointer',
                }}
              />
            </div>

            {/* Include MM toggle */}
            <label
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                cursor: 'pointer',
                fontSize: 11,
                color: 'var(--text-secondary)',
                marginBottom: 8,
              }}
            >
              <input
                type="checkbox"
                checked={config.include_mm}
                onChange={(e) =>
                  onConfigChange({ ...config, include_mm: e.target.checked })
                }
                style={{ accentColor: '#ffaa00' }}
              />
              Include multi-messenger
            </label>

            {/* Divider */}
            <div style={{ borderTop: '1px solid var(--border)', marginBottom: 8 }} />

            {/* Summary stats bar */}
            <div
              style={{
                display: 'flex',
                gap: 6,
                marginBottom: 8,
                flexWrap: 'wrap',
              }}
            >
              <StatChip
                label="Total"
                value={stats.total}
                color="#ffaa00"
                active={filterMode === 'all'}
                onClick={() => setFilterMode('all')}
              />
              <StatChip
                label="Unexplained"
                value={stats.unexplained}
                color="#ff4444"
                active={filterMode === 'unexplained'}
                onClick={() => setFilterMode('unexplained')}
              />
              {stats.partial > 0 && (
                <StatChip
                  label="Partial"
                  value={stats.partial}
                  color="#ffaa00"
                  active={filterMode === 'partial'}
                  onClick={() => setFilterMode('partial')}
                />
              )}
              <StatChip
                label="Explained"
                value={stats.explained}
                color="#44cc44"
                active={filterMode === 'explained'}
                onClick={() => setFilterMode('explained')}
              />
              {stats.fdr > 0 && (
                <span
                  className="mono"
                  style={{
                    fontSize: 9,
                    padding: '2px 6px',
                    borderRadius: 4,
                    background: 'rgba(255,255,255,0.05)',
                    color: 'var(--text-dim)',
                  }}
                >
                  {stats.fdr} FDR
                </span>
              )}
            </div>

            {/* Sort controls */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 8 }}>
              <span style={{ fontSize: 9, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Sort:
              </span>
              <SortButton
                label="Signals"
                active={sortMode === 'signals'}
                onClick={() => setSortMode('signals')}
              />
              <SortButton
                label="Score"
                active={sortMode === 'score'}
                onClick={() => setSortMode('score')}
              />
              <SortButton
                label="Unex"
                active={sortMode === 'unexplainability'}
                onClick={() => setSortMode('unexplainability')}
              />
              <SortButton
                label="Dist"
                active={sortMode === 'distance'}
                onClick={() => setSortMode('distance')}
              />
            </div>

            {/* Channel filter chips */}
            {stats.channels.length > 0 && (
              <div style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 9, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>
                  Channel Filter
                  {channelFilter.size > 0 && (
                    <button
                      onClick={() => setChannelFilter(new Set())}
                      style={{
                        marginLeft: 6, padding: '0 4px', background: 'rgba(255,170,0,0.15)',
                        border: '1px solid rgba(255,170,0,0.3)', borderRadius: 3,
                        color: '#ffaa00', fontSize: 8, cursor: 'pointer', lineHeight: '14px',
                      }}
                    >
                      clear
                    </button>
                  )}
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                  {stats.channels.map((ch) => {
                    const isActive = channelFilter.has(ch);
                    const chColor = getChannelColor(ch);
                    const count = stats.channelCounts.get(ch) ?? 0;
                    return (
                      <button
                        key={ch}
                        onClick={() => {
                          const next = new Set(channelFilter);
                          if (isActive) next.delete(ch);
                          else next.add(ch);
                          setChannelFilter(next);
                        }}
                        style={{
                          padding: '1px 5px',
                          borderRadius: 3,
                          fontSize: 8,
                          fontWeight: isActive ? 700 : 400,
                          background: isActive ? `${chColor}22` : 'rgba(255,255,255,0.03)',
                          color: isActive ? chColor : 'var(--text-dim)',
                          border: isActive ? `1px solid ${chColor}44` : '1px solid transparent',
                          cursor: 'pointer',
                          transition: 'all 0.12s',
                          lineHeight: '16px',
                        }}
                      >
                        {ch}
                        <span style={{ opacity: 0.6, marginLeft: 2, fontSize: 7 }}>{count}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Campaign filter */}
            {stats.campaigns.length > 1 && (
              <div style={{ marginBottom: 4 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <span style={{ fontSize: 9, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    Campaign:
                  </span>
                  <select
                    value={campaignFilter}
                    onChange={(e) => setCampaignFilter(e.target.value)}
                    style={{
                      padding: '2px 6px',
                      fontSize: 9,
                      fontFamily: 'var(--font-mono)',
                      background: 'rgba(255,255,255,0.06)',
                      color: campaignFilter === 'all' ? 'var(--text-dim)' : 'var(--text-bright)',
                      border: '1px solid var(--border)',
                      borderRadius: 4,
                      cursor: 'pointer',
                      outline: 'none',
                    }}
                  >
                    <option value="all">All campaigns</option>
                    {stats.campaigns.map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </div>
              </div>
            )}
          </div>

          {/* Zone list (scrollable) */}
          <div style={{ overflow: 'auto', flex: 1, padding: '0 12px 10px' }}>
            {computing ? (
              <div
                style={{
                  padding: 16,
                  textAlign: 'center',
                  fontSize: 11,
                  color: 'var(--text-dim)',
                }}
              >
                Computing convergence...
              </div>
            ) : sortedZones.length === 0 ? (
              <div
                style={{
                  padding: 16,
                  textAlign: 'center',
                  fontSize: 11,
                  color: 'var(--text-dim)',
                }}
              >
                {filterMode !== 'all'
                  ? `No ${filterMode} zones found with current settings.`
                  : 'No convergence zones found.'}
                <br />
                <span style={{ fontSize: 10, marginTop: 4, display: 'block' }}>
                  {filterMode !== 'all'
                    ? 'Try changing the filter or adjusting parameters.'
                    : 'Try increasing the radius or decreasing min channels.'}
                </span>
              </div>
            ) : (
              <div>
                <div
                  style={{
                    fontSize: 10,
                    color: 'var(--text-dim)',
                    textTransform: 'uppercase',
                    letterSpacing: '1px',
                    marginBottom: 6,
                  }}
                >
                  {filterMode === 'all'
                    ? `Zones (${sortedZones.length})`
                    : `${filterMode} (${sortedZones.length}/${zones.length})`}
                </div>
                {sortedZones.slice(0, 80).map((zone, i) => (
                  <ZoneCard
                    key={zone.id}
                    zone={zone}
                    index={i}
                    onClick={() => zone.anchor_target && onGotoZone(zone.anchor_target)}
                  />
                ))}
                {sortedZones.length > 80 && (
                  <div style={{ padding: 8, textAlign: 'center', fontSize: 10, color: 'var(--text-dim)' }}>
                    +{sortedZones.length - 80} more zones...
                  </div>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

/** Individual zone card with unexplainability display */
function ZoneCard({
  zone,
  index,
  onClick,
}: {
  zone: ConvergenceZone;
  index: number;
  onClick: () => void;
}) {
  const status = getZoneStatus(zone);
  const isTop = index < 3;

  return (
    <button
      onClick={onClick}
      style={{
        display: 'block',
        width: '100%',
        padding: '7px 10px',
        marginBottom: 3,
        background: isTop
          ? 'rgba(255,170,0,0.06)'
          : 'rgba(255,255,255,0.02)',
        border: isTop
          ? '1px solid rgba(255,170,0,0.2)'
          : '1px solid transparent',
        borderRadius: 6,
        cursor: 'pointer',
        textAlign: 'left',
        color: 'var(--text-primary)',
        transition: 'all 0.15s',
      }}
    >
      {/* Row 1: ID + signal count + status badge */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 3,
        }}
      >
        <span
          className="mono"
          style={{
            fontSize: 10,
            fontWeight: 600,
            color: 'var(--text-bright)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            flex: 1,
            minWidth: 0,
          }}
        >
          {zone.id}
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4, flexShrink: 0 }}>
          <span
            className="mono"
            style={{
              fontSize: 10,
              fontWeight: 700,
              color: convergenceColor(zone.n_channels),
              textShadow:
                zone.n_channels >= 4 ? '0 0 4px rgba(255,200,50,0.5)' : 'none',
            }}
          >
            {zone.n_channels}ch
          </span>
          <span
            style={{
              fontSize: 7,
              fontWeight: 700,
              padding: '1px 4px',
              borderRadius: 3,
              background: status.bgColor,
              color: status.color,
              border: `1px solid ${status.borderColor}`,
              textTransform: 'uppercase',
              letterSpacing: '0.3px',
              whiteSpace: 'nowrap',
            }}
          >
            {status.label}
          </span>
        </div>
      </div>

      {/* Row 2: Channel badges */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 2,
          marginBottom: 3,
        }}
      >
        {zone.channels.map((ch) => (
          <span
            key={ch}
            style={{
              padding: '0px 4px',
              borderRadius: 3,
              fontSize: 8,
              background: 'rgba(255,255,255,0.05)',
              color: getChannelColor(ch),
              border: `1px solid ${getChannelColor(ch)}22`,
              lineHeight: '14px',
            }}
          >
            {ch}
          </span>
        ))}
      </div>

      {/* Row 3: Score + Unexplainability bar + coordinates */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}
      >
        {zone.score != null && (
          <span className="mono" style={{ fontSize: 9, color: 'var(--text-dim)' }}>
            S:{zone.score.toFixed(3)}
          </span>
        )}
        {zone.anchor_target?.distance_pc != null && (
          <span className="mono" style={{ fontSize: 9, color: 'var(--text-dim)' }}>
            {zone.anchor_target.distance_pc.toFixed(0)}pc
          </span>
        )}
        {zone.unexplainability != null && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 3, flex: 1 }}>
            {/* Mini bar */}
            <div
              style={{
                flex: 1,
                height: 3,
                background: 'rgba(255,255,255,0.06)',
                borderRadius: 2,
                overflow: 'hidden',
                maxWidth: 50,
              }}
            >
              <div
                style={{
                  width: `${Math.min(100, zone.unexplainability * 100)}%`,
                  height: '100%',
                  background: unexColor(zone.unexplainability),
                  borderRadius: 2,
                }}
              />
            </div>
            <span
              className="mono"
              style={{
                fontSize: 9,
                color: unexColor(zone.unexplainability),
                fontWeight: zone.unexplainability >= 0.5 ? 700 : 400,
              }}
            >
              {zone.unexplainability.toFixed(2)}
            </span>
          </div>
        )}
        {zone.fdr_significant && (
          <span
            style={{
              fontSize: 7,
              fontWeight: 700,
              padding: '0px 3px',
              borderRadius: 2,
              background: 'rgba(255,255,255,0.1)',
              color: '#ffffff',
              lineHeight: '12px',
            }}
          >
            FDR
          </span>
        )}
        <span
          className="mono"
          style={{ fontSize: 8, color: 'var(--text-dim)', marginLeft: 'auto' }}
        >
          {zone.ra.toFixed(1)}° {zone.dec >= 0 ? '+' : ''}{zone.dec.toFixed(1)}°
        </span>
      </div>
    </button>
  );
}

/** Stat chip — clickable filter */
function StatChip({
  label,
  value,
  color,
  active,
  onClick,
}: {
  label: string;
  value: number;
  color: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 4,
        padding: '2px 7px',
        borderRadius: 4,
        fontSize: 9,
        fontWeight: active ? 700 : 500,
        background: active ? `${color}22` : 'rgba(255,255,255,0.03)',
        color: active ? color : 'var(--text-dim)',
        border: active ? `1px solid ${color}44` : '1px solid transparent',
        cursor: 'pointer',
        transition: 'all 0.15s',
      }}
    >
      <span
        style={{
          width: 5,
          height: 5,
          borderRadius: '50%',
          background: active ? color : 'transparent',
          border: `1.5px solid ${color}`,
          flexShrink: 0,
        }}
      />
      <span className="mono">{value}</span>
      <span>{label}</span>
    </button>
  );
}

/** Sort mode button */
function SortButton({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '2px 6px',
        borderRadius: 3,
        fontSize: 9,
        fontWeight: active ? 600 : 400,
        background: active ? 'rgba(255,170,0,0.15)' : 'transparent',
        color: active ? '#ffaa00' : 'var(--text-dim)',
        border: active ? '1px solid rgba(255,170,0,0.3)' : '1px solid transparent',
        cursor: 'pointer',
        transition: 'all 0.15s',
      }}
    >
      {label}
    </button>
  );
}

/** Map channel name to a display color */
function getChannelColor(ch: string): string {
  const map: Record<string, string> = {
    'IR Excess': '#ff9800',
    'IR Variability': '#ffb74d',
    'PM Anomaly': '#e040fb',
    'HZ Prior': '#2196f3',
    'Transit': '#4caf50',
    'Radio': '#ff5722',
    'UV Anomaly': '#7c4dff',
    'HR Anomaly': '#e91e63',
    'Abundance': '#8bc34a',
    'Gaia Phot': '#00bcd4',
    'Fermi Unid': '#ffffff',
    'IceCube': '#9c27b0',
    'FRB': '#f44336',
    'NANOGrav': '#b0bec5',
  };
  return map[ch] || '#888888';
}
