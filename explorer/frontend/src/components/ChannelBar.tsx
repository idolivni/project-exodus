/**
 * EXODUS Galaxy Explorer — Channel Bar (top navigation)
 *
 * Horizontal bar with search, detection channel toggles,
 * multi-messenger toggles, distance filter, and target counts.
 * Replaces the old LayerPanel sidebar to free up left-side space.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import type { LayerConfig, ScoredTarget } from '../types';
import { useSearch } from '../hooks/useApi';

interface CatalogCounts {
  fermi_all: number;
  fermi_unid: number;
  icecube: number;
  frb: number;
  pulsars: number;
}

interface ChannelBarProps {
  layers: LayerConfig[];
  onToggle: (id: string) => void;
  onOpacityChange: (id: string, opacity: number) => void;
  scoredCount: number;
  gatheredCount: number;
  catalogCounts?: CatalogCounts;
  onGotoTarget?: (target: ScoredTarget) => void;
  distanceRange: [number, number];
  onDistanceRangeChange: (range: [number, number]) => void;
}

// Detection channel layer IDs with short labels
const DETECTION_CHANNELS = [
  { id: 'exodus_score', short: 'Score' },
  { id: 'ir_excess', short: 'IR' },
  { id: 'ruwe_anomaly', short: 'RUWE' },
  { id: 'pm_discrepancy', short: 'PM' },
  { id: 'hz_prior', short: 'HZ' },
];

// Multi-messenger layer IDs with short labels and count keys
const MM_CHANNELS = [
  { id: 'fermi_unid', short: 'Fermi', countKey: 'fermi_unid' as keyof CatalogCounts },
  { id: 'icecube', short: 'ICube', countKey: 'icecube' as keyof CatalogCounts },
  { id: 'frb', short: 'FRB', countKey: 'frb' as keyof CatalogCounts },
  { id: 'pulsars', short: 'PSR', countKey: 'pulsars' as keyof CatalogCounts },
];

/** Format catalog count */
function fmtCount(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

/** Distance preset stops for the slider */
const DIST_STOPS = [0, 20, 50, 100, 200, 500, 1000, 2000, 5000];
function distFromSlider(val: number): number {
  const idx = Math.floor(val);
  const frac = val - idx;
  if (idx >= DIST_STOPS.length - 1) return DIST_STOPS[DIST_STOPS.length - 1];
  return DIST_STOPS[idx] + frac * (DIST_STOPS[idx + 1] - DIST_STOPS[idx]);
}
function distToSlider(d: number): number {
  for (let i = 0; i < DIST_STOPS.length - 1; i++) {
    if (d <= DIST_STOPS[i + 1]) {
      return i + (d - DIST_STOPS[i]) / (DIST_STOPS[i + 1] - DIST_STOPS[i]);
    }
  }
  return DIST_STOPS.length - 1;
}

export default function ChannelBar({
  layers,
  onToggle,
  catalogCounts,
  onGotoTarget,
  scoredCount,
  gatheredCount,
  distanceRange,
  onDistanceRangeChange,
}: ChannelBarProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchFocused, setSearchFocused] = useState(false);
  const { results: searchResults, loading: searchLoading, search } = useSearch();

  // Debounced search
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();
  const handleSearchInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const q = e.target.value;
      setSearchQuery(q);
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => search(q), 300);
    },
    [search]
  );
  useEffect(() => () => { if (debounceRef.current) clearTimeout(debounceRef.current); }, []);

  const handleSearchSelect = useCallback(
    (t: ScoredTarget) => {
      if (onGotoTarget) onGotoTarget(t);
      setSearchQuery('');
      setSearchFocused(false);
    },
    [onGotoTarget]
  );

  const getLayer = (id: string) => layers.find(l => l.id === id);

  // Distance slider state (local for smooth dragging)
  const [localDistMin, setLocalDistMin] = useState(distToSlider(distanceRange[0]));
  const [localDistMax, setLocalDistMax] = useState(distToSlider(distanceRange[1] === Infinity ? 5000 : distanceRange[1]));

  const commitDistance = useCallback(() => {
    const min = Math.round(distFromSlider(localDistMin));
    const max = Math.round(distFromSlider(localDistMax));
    onDistanceRangeChange([min, max >= 5000 ? Infinity : max]);
  }, [localDistMin, localDistMax, onDistanceRangeChange]);

  const isDistFiltered = distanceRange[0] > 0 || distanceRange[1] < Infinity;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        height: 'var(--topbar-height, 44px)',
        background: 'var(--bg-panel)',
        borderBottom: '1px solid var(--border)',
        backdropFilter: 'blur(12px)',
        zIndex: 100,
        display: 'flex',
        alignItems: 'center',
        gap: 0,
        padding: '0 12px',
        fontFamily: 'var(--font-body)',
      }}
    >
      {/* ── Brand ─────────────────────────────── */}
      <span
        style={{
          fontSize: 13,
          fontWeight: 700,
          color: 'var(--gold)',
          letterSpacing: '2px',
          marginRight: 12,
          flexShrink: 0,
          userSelect: 'none',
        }}
      >
        EXODUS
      </span>

      {/* ── Search ────────────────────────────── */}
      <div style={{ position: 'relative', width: 180, flexShrink: 0, marginRight: 8 }}>
        <input
          type="text"
          placeholder="🔍 Search targets..."
          value={searchQuery}
          onChange={handleSearchInput}
          onFocus={() => setSearchFocused(true)}
          onBlur={() => setTimeout(() => setSearchFocused(false), 200)}
          style={{
            width: '100%',
            padding: '4px 8px',
            background: 'rgba(255,255,255,0.06)',
            border: '1px solid var(--border)',
            borderRadius: 5,
            color: 'var(--text-primary)',
            fontSize: 11,
            fontFamily: 'var(--font-mono)',
            outline: 'none',
            height: 28,
          }}
        />
        {/* Search dropdown */}
        {searchFocused && searchQuery.length >= 2 && (
          <div
            style={{
              position: 'absolute',
              top: '100%',
              left: 0,
              width: 300,
              background: 'var(--bg-card)',
              border: '1px solid var(--border-active)',
              borderRadius: 6,
              marginTop: 4,
              maxHeight: 250,
              overflow: 'auto',
              zIndex: 300,
              boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
            }}
          >
            {searchLoading ? (
              <div style={{ padding: 8, fontSize: 10, color: 'var(--text-dim)' }}>Searching...</div>
            ) : searchResults.length === 0 ? (
              <div style={{ padding: 8, fontSize: 10, color: 'var(--text-dim)' }}>No results</div>
            ) : (
              searchResults.map((t) => (
                <button
                  key={t.id}
                  onMouseDown={(e) => { e.preventDefault(); handleSearchSelect(t); }}
                  style={{
                    display: 'block',
                    width: '100%',
                    padding: '6px 10px',
                    background: 'none',
                    border: 'none',
                    borderBottom: '1px solid var(--border)',
                    color: 'var(--text-primary)',
                    fontSize: 10,
                    fontFamily: 'var(--font-mono)',
                    textAlign: 'left',
                    cursor: 'pointer',
                  }}
                >
                  <div style={{ fontWeight: 500 }}>{t.id}</div>
                  <div style={{ fontSize: 9, color: 'var(--text-dim)', marginTop: 1 }}>
                    {t.total_score != null ? `Score: ${t.total_score.toFixed(4)}` : t.status ?? 'pending'}
                    {t.distance_pc != null && ` · ${t.distance_pc.toFixed(0)} pc`}
                    {t.n_active_channels != null && ` · ${t.n_active_channels}ch`}
                  </div>
                </button>
              ))
            )}
          </div>
        )}
      </div>

      {/* ── Separator ──────────────────────────── */}
      <Separator />

      {/* ── Detection Channels ─────────────────── */}
      <span style={{ fontSize: 8, color: 'var(--text-dim)', letterSpacing: '0.5px', marginRight: 6, textTransform: 'uppercase', flexShrink: 0 }}>
        Channels
      </span>
      <div style={{ display: 'flex', gap: 3, flexShrink: 0 }}>
        {DETECTION_CHANNELS.map(({ id, short }) => {
          const layer = getLayer(id);
          if (!layer) return null;
          return (
            <TogglePill
              key={id}
              label={short}
              color={layer.color}
              active={layer.visible}
              onClick={() => onToggle(id)}
            />
          );
        })}
      </div>

      {/* ── Separator ──────────────────────────── */}
      <Separator />

      {/* ── Multi-Messenger ────────────────────── */}
      <span style={{ fontSize: 8, color: 'var(--text-dim)', letterSpacing: '0.5px', marginRight: 6, textTransform: 'uppercase', flexShrink: 0 }}>
        MM
      </span>
      <div style={{ display: 'flex', gap: 3, flexShrink: 0 }}>
        {MM_CHANNELS.map(({ id, short, countKey }) => {
          const layer = getLayer(id);
          if (!layer) return null;
          const count = catalogCounts ? catalogCounts[countKey] : 0;
          return (
            <TogglePill
              key={id}
              label={short}
              color={layer.color}
              active={layer.visible}
              onClick={() => onToggle(id)}
              badge={count > 0 ? fmtCount(count) : undefined}
            />
          );
        })}
      </div>

      {/* ── Separator ──────────────────────────── */}
      <Separator />

      {/* ── Distance Filter ────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0, minWidth: 180 }}>
        <span style={{ fontSize: 8, color: isDistFiltered ? 'var(--gold)' : 'var(--text-dim)', letterSpacing: '0.5px', textTransform: 'uppercase' }}>
          Dist
        </span>
        <span className="mono" style={{ fontSize: 9, color: 'var(--text-secondary)', minWidth: 32, textAlign: 'right' }}>
          {Math.round(distFromSlider(localDistMin))}
        </span>
        <div style={{ position: 'relative', width: 80, height: 20 }}>
          {/* Min slider */}
          <input
            type="range"
            min={0}
            max={DIST_STOPS.length - 1}
            step={0.1}
            value={localDistMin}
            onChange={(e) => setLocalDistMin(Math.min(Number(e.target.value), localDistMax - 0.1))}
            onMouseUp={commitDistance}
            onTouchEnd={commitDistance}
            style={{
              position: 'absolute', top: 0, left: 0, width: '100%',
              height: 20, appearance: 'none', background: 'transparent',
              pointerEvents: 'all', zIndex: 2, cursor: 'pointer',
            }}
          />
          {/* Max slider */}
          <input
            type="range"
            min={0}
            max={DIST_STOPS.length - 1}
            step={0.1}
            value={localDistMax}
            onChange={(e) => setLocalDistMax(Math.max(Number(e.target.value), localDistMin + 0.1))}
            onMouseUp={commitDistance}
            onTouchEnd={commitDistance}
            style={{
              position: 'absolute', top: 0, left: 0, width: '100%',
              height: 20, appearance: 'none', background: 'transparent',
              pointerEvents: 'all', zIndex: 1, cursor: 'pointer',
            }}
          />
          {/* Track visualization */}
          <div style={{
            position: 'absolute', top: 9, left: 0, right: 0, height: 2,
            background: 'rgba(255,255,255,0.1)', borderRadius: 1,
          }}>
            <div style={{
              position: 'absolute',
              left: `${(localDistMin / (DIST_STOPS.length - 1)) * 100}%`,
              right: `${100 - (localDistMax / (DIST_STOPS.length - 1)) * 100}%`,
              top: 0, bottom: 0,
              background: isDistFiltered ? 'var(--gold)' : 'rgba(255,255,255,0.3)',
              borderRadius: 1,
            }} />
          </div>
        </div>
        <span className="mono" style={{ fontSize: 9, color: 'var(--text-secondary)', minWidth: 36 }}>
          {distFromSlider(localDistMax) >= 5000 ? '∞' : Math.round(distFromSlider(localDistMax))}
          <span style={{ fontSize: 7, marginLeft: 1 }}>pc</span>
        </span>
        {isDistFiltered && (
          <button
            onClick={() => {
              setLocalDistMin(0);
              setLocalDistMax(DIST_STOPS.length - 1);
              onDistanceRangeChange([0, Infinity]);
            }}
            style={{
              padding: '1px 4px', background: 'rgba(255,215,0,0.15)',
              border: '1px solid rgba(255,215,0,0.3)', borderRadius: 3,
              color: 'var(--gold)', fontSize: 8, cursor: 'pointer',
            }}
          >
            ×
          </button>
        )}
      </div>

      {/* ── Separator ──────────────────────────── */}
      <Separator />

      {/* ── Survey Quick-Switch ──────────────────── */}
      <SurveySwitch />

      {/* ── Spacer ─────────────────────────────── */}
      <div style={{ flex: 1 }} />

      {/* ── Counts ─────────────────────────────── */}
      <div className="mono" style={{ fontSize: 9, color: 'var(--text-dim)', flexShrink: 0 }}>
        <span style={{ color: 'var(--text-secondary)' }}>{scoredCount}</span>
        <span> scored</span>
        {gatheredCount > 0 && (
          <>
            <span style={{ margin: '0 3px' }}>·</span>
            <span style={{ color: 'var(--text-secondary)' }}>{gatheredCount}</span>
            <span> gathering</span>
          </>
        )}
      </div>
    </div>
  );
}

/** Small vertical separator */
function Separator() {
  return (
    <div
      style={{
        width: 1,
        height: 20,
        background: 'var(--border)',
        margin: '0 10px',
        flexShrink: 0,
      }}
    />
  );
}

/** Survey quick-switch — lets astronomer toggle base image survey in 1 click */
const SURVEYS = [
  { id: 'P/DSS2/color', label: 'DSS2', desc: 'Optical (DSS2 color)' },
  { id: 'P/2MASS/color', label: '2MASS', desc: 'Near-IR (J/H/K)' },
  { id: 'P/allWISE/color', label: 'WISE', desc: 'Mid-IR (W1-W4)' },
  { id: 'P/GALEXGR6_7/AIS/color', label: 'GALEX', desc: 'UV (NUV/FUV)' },
] as const;

function SurveySwitch() {
  const [current, setCurrent] = useState('P/DSS2/color');

  const handleSwitch = (surveyId: string) => {
    const al = (window as unknown as Record<string, unknown>).__aladin as
      | { setImageSurvey?: (s: string) => void }
      | undefined;
    if (al?.setImageSurvey) {
      al.setImageSurvey(surveyId);
      setCurrent(surveyId);
    }
  };

  return (
    <div style={{ display: 'flex', gap: 2, flexShrink: 0 }}>
      {SURVEYS.map(({ id, label, desc }) => {
        const active = current === id;
        return (
          <button
            key={id}
            title={desc}
            onClick={() => handleSwitch(id)}
            style={{
              padding: '2px 6px',
              height: 22,
              borderRadius: 4,
              fontSize: 9,
              fontWeight: active ? 600 : 400,
              background: active ? 'rgba(100,200,255,0.15)' : 'transparent',
              color: active ? '#64c8ff' : 'var(--text-dim)',
              border: active ? '1px solid rgba(100,200,255,0.3)' : '1px solid transparent',
              cursor: 'pointer',
              transition: 'all 0.12s',
              flexShrink: 0,
            }}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}

/** Toggle pill button for channel/MM layers */
function TogglePill({
  label,
  color,
  active,
  onClick,
  badge,
}: {
  label: string;
  color: string;
  active: boolean;
  onClick: () => void;
  badge?: string;
}) {
  return (
    <button
      onClick={onClick}
      title={`Toggle ${label}`}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 4,
        padding: '3px 8px',
        height: 26,
        background: active ? `${color}18` : 'transparent',
        border: `1px solid ${active ? `${color}60` : 'var(--border)'}`,
        borderRadius: 13,
        cursor: 'pointer',
        color: active ? 'var(--text-bright)' : 'var(--text-dim)',
        fontSize: 10,
        fontWeight: active ? 500 : 400,
        transition: 'all 0.15s',
        flexShrink: 0,
      }}
    >
      <span
        style={{
          width: 7,
          height: 7,
          borderRadius: '50%',
          background: active ? color : 'transparent',
          border: `1.5px solid ${color}`,
          opacity: active ? 1 : 0.4,
          transition: 'all 0.15s',
        }}
      />
      <span>{label}</span>
      {badge && (
        <span
          className="mono"
          style={{
            fontSize: 8,
            color: active ? 'var(--text-secondary)' : 'var(--text-dim)',
            opacity: 0.7,
          }}
        >
          {badge}
        </span>
      )}
    </button>
  );
}
