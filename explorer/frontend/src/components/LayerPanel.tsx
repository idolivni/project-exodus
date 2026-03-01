/**
 * EXODUS Galaxy Explorer — Layer Panel (sidebar)
 *
 * Phase 3.5: Left sidebar positioned BELOW Aladin controls with:
 * - Collapsible panel with compact header
 * - Target search
 * - Detection channel layer toggles with opacity sliders
 * - Multi-messenger overlay toggles with catalog counts
 * - Score legend + markers
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

interface LayerPanelProps {
  layers: LayerConfig[];
  onToggle: (id: string) => void;
  onOpacityChange: (id: string, opacity: number) => void;
  targetCount: number;
  scoredCount: number;
  gatheredCount: number;
  catalogCounts?: CatalogCounts;
  onGotoTarget?: (target: ScoredTarget) => void;
}

// Detection channel IDs
const DETECTION_LAYERS = ['exodus_score', 'ir_excess', 'ruwe_anomaly', 'pm_discrepancy', 'hz_prior'];
// Multi-messenger catalog IDs
const MM_LAYERS = ['fermi_unid', 'fermi_all', 'icecube', 'frb', 'pulsars'];

/** Format catalog count for display */
function formatCount(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

export default function LayerPanel({
  layers,
  onToggle,
  onOpacityChange,
  targetCount,
  scoredCount,
  gatheredCount,
  catalogCounts,
  onGotoTarget,
}: LayerPanelProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedSlider, setExpandedSlider] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState(false);
  const [legendOpen, setLegendOpen] = useState(false);
  const { results: searchResults, loading: searchLoading, search } = useSearch();

  // Debounced search — waits 300ms after user stops typing
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
  // Cleanup debounce on unmount
  useEffect(() => () => { if (debounceRef.current) clearTimeout(debounceRef.current); }, []);

  const handleSearchSelect = useCallback(
    (t: ScoredTarget) => {
      if (onGotoTarget) onGotoTarget(t);
      setSearchQuery('');
    },
    [onGotoTarget]
  );

  const detectionLayers = layers.filter(l => DETECTION_LAYERS.includes(l.id));
  const mmLayers = layers.filter(l => MM_LAYERS.includes(l.id));

  /** Count string for a multi-messenger layer */
  const getCatalogCount = (layerId: string): string | null => {
    if (!catalogCounts) return null;
    const countMap: Record<string, number> = {
      fermi_all: catalogCounts.fermi_all,
      fermi_unid: catalogCounts.fermi_unid,
      icecube: catalogCounts.icecube,
      frb: catalogCounts.frb,
      pulsars: catalogCounts.pulsars,
    };
    const n = countMap[layerId];
    return n !== undefined ? formatCount(n) : null;
  };

  const renderLayerRow = (layer: LayerConfig, showCount = false) => {
    const count = showCount ? getCatalogCount(layer.id) : null;
    const isExpanded = expandedSlider === layer.id;

    return (
      <div key={layer.id} style={{ marginBottom: 1 }}>
        {/* Layer toggle row */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            width: '100%',
          }}
        >
          <button
            onClick={() => onToggle(layer.id)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              flex: 1,
              padding: '4px 6px',
              background: layer.visible ? 'rgba(255,255,255,0.06)' : 'transparent',
              border: 'none',
              borderRadius: 4,
              cursor: 'pointer',
              color: layer.visible ? 'var(--text-primary)' : 'var(--text-dim)',
              fontSize: 11,
              textAlign: 'left',
              transition: 'all 0.15s',
            }}
          >
            {/* Color indicator */}
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                background: layer.visible ? layer.color : 'transparent',
                border: `2px solid ${layer.color}`,
                flexShrink: 0,
                opacity: layer.visible ? 1 : 0.4,
                transition: 'all 0.2s',
              }}
            />
            <span style={{ flex: 1 }}>{layer.label}</span>
            {count && (
              <span
                className="mono"
                style={{
                  fontSize: 9,
                  color: 'var(--text-dim)',
                }}
              >
                {count}
              </span>
            )}
          </button>

          {/* Opacity toggle button */}
          {layer.visible && (
            <button
              onClick={() => setExpandedSlider(isExpanded ? null : layer.id)}
              title={`Opacity: ${Math.round(layer.opacity * 100)}%`}
              style={{
                padding: '2px 4px',
                background: isExpanded ? 'rgba(255,255,255,0.1)' : 'transparent',
                border: '1px solid var(--border)',
                borderRadius: 3,
                cursor: 'pointer',
                color: 'var(--text-dim)',
                fontSize: 8,
                fontFamily: 'var(--font-mono)',
                minWidth: 28,
                textAlign: 'center',
                transition: 'all 0.15s',
              }}
            >
              {Math.round(layer.opacity * 100)}%
            </button>
          )}
        </div>

        {/* Opacity slider (expanded) */}
        {isExpanded && layer.visible && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '2px 6px 4px 22px',
            }}
          >
            <input
              type="range"
              min={0}
              max={100}
              value={Math.round(layer.opacity * 100)}
              onChange={(e) => onOpacityChange(layer.id, Number(e.target.value) / 100)}
              style={{
                flex: 1,
                height: 2,
                accentColor: layer.color,
                cursor: 'pointer',
              }}
            />
          </div>
        )}
      </div>
    );
  };

  return (
    <div
      className="glass-panel"
      style={{
        position: 'absolute',
        top: 56,
        left: 12,
        width: collapsed ? 42 : 240,
        maxHeight: 'calc(100vh - 100px)',
        overflow: collapsed ? 'hidden' : 'auto',
        zIndex: 100,
        padding: collapsed ? '8px 10px' : '12px',
        transition: 'width 0.2s ease',
      }}
    >
      {/* Header — always visible */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          cursor: 'pointer',
          marginBottom: collapsed ? 0 : 8,
        }}
        onClick={() => setCollapsed(!collapsed)}
      >
        {collapsed ? (
          <span style={{ fontSize: 14, color: 'var(--gold)', lineHeight: 1 }}>&#9776;</span>
        ) : (
          <>
            <span
              style={{
                fontSize: 13,
                fontWeight: 700,
                color: 'var(--gold)',
                letterSpacing: '2px',
                flex: 1,
              }}
            >
              EXODUS
            </span>
            <span className="mono" style={{ fontSize: 9, color: 'var(--text-dim)' }}>
              {scoredCount}{gatheredCount > 0 ? `+${gatheredCount}` : ''}/{targetCount}
            </span>
            <span style={{ fontSize: 9, color: 'var(--text-dim)' }}>&#9664;</span>
          </>
        )}
      </div>

      {/* Collapsed — nothing more */}
      {collapsed ? null : (
        <>
          {/* Search bar */}
          <div style={{ marginBottom: 10, position: 'relative' }}>
            <input
              type="text"
              placeholder="Search targets..."
              value={searchQuery}
              onChange={handleSearchInput}
              style={{
                width: '100%',
                padding: '5px 8px',
                background: 'rgba(255,255,255,0.06)',
                border: '1px solid var(--border)',
                borderRadius: 5,
                color: 'var(--text-primary)',
                fontSize: 11,
                fontFamily: 'var(--font-mono)',
                outline: 'none',
              }}
            />
            {/* Search results dropdown */}
            {searchQuery.length >= 2 && (
              <div
                style={{
                  position: 'absolute',
                  top: '100%',
                  left: 0,
                  right: 0,
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border)',
                  borderRadius: 6,
                  marginTop: 4,
                  maxHeight: 200,
                  overflow: 'auto',
                  zIndex: 200,
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
                      onClick={() => handleSearchSelect(t)}
                      style={{
                        display: 'block',
                        width: '100%',
                        padding: '5px 8px',
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
                        {t.distance_pc != null && ` | ${t.distance_pc.toFixed(1)} pc`}
                      </div>
                    </button>
                  ))
                )}
              </div>
            )}
          </div>

          {/* Detection Channels section */}
          <div
            style={{
              fontSize: 9,
              color: 'var(--text-dim)',
              textTransform: 'uppercase',
              marginBottom: 4,
              letterSpacing: '1px',
            }}
          >
            Channels
          </div>
          {detectionLayers.map((layer) => renderLayerRow(layer, false))}

          {/* Multi-Messenger section */}
          <div
            style={{
              fontSize: 9,
              color: 'var(--text-dim)',
              textTransform: 'uppercase',
              marginTop: 10,
              marginBottom: 4,
              letterSpacing: '1px',
              borderTop: '1px solid var(--border)',
              paddingTop: 8,
            }}
          >
            Multi-Messenger
          </div>
          {mmLayers.map((layer) => renderLayerRow(layer, true))}

          {/* Collapsible Legend */}
          <div style={{ marginTop: 10, borderTop: '1px solid var(--border)', paddingTop: 8 }}>
            <div
              onClick={() => setLegendOpen(!legendOpen)}
              style={{
                fontSize: 9,
                color: 'var(--text-dim)',
                textTransform: 'uppercase',
                letterSpacing: '1px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 4,
              }}
            >
              <span>{legendOpen ? '▾' : '▸'}</span>
              <span>Legend & Markers</span>
            </div>

            {legendOpen && (
              <div style={{ marginTop: 6 }}>
                {/* Score Key */}
                {[
                  { color: '#ff4444', label: 'High (>0.5)' },
                  { color: '#ff9800', label: 'Medium (0.2-0.5)' },
                  { color: '#ffd700', label: 'Low (0.05-0.2)' },
                  { color: '#4488ff', label: 'Baseline (<0.05)' },
                  { color: '#555588', label: 'Gathering...' },
                ].map(({ color, label }) => (
                  <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 2 }}>
                    <span
                      style={{
                        width: 7,
                        height: 7,
                        borderRadius: '50%',
                        background: color,
                        flexShrink: 0,
                      }}
                    />
                    <span style={{ fontSize: 10, color: 'var(--text-secondary)' }}>{label}</span>
                  </div>
                ))}

                {/* Markers */}
                <div style={{ marginTop: 6, paddingTop: 6, borderTop: '1px solid var(--border)' }}>
                  {[
                    { shape: 'diamond', color: '#ffd700', glow: true, label: 'Prime Candidate' },
                    { shape: 'diamond', color: '#ffffff', glow: true, label: 'FDR Significant' },
                    { shape: 'square', color: '#ff4444', glow: false, label: 'Investigate' },
                    { shape: 'circle', color: '#44cc44', glow: false, label: 'Explained' },
                    { shape: 'ring', color: '#ffaa00', glow: false, label: 'Convergence Zone' },
                  ].map(({ shape, color, glow, label }) => (
                    <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 2 }}>
                      <span
                        style={{
                          width: 9,
                          height: 9,
                          flexShrink: 0,
                          background: shape === 'ring' ? 'transparent' : color,
                          border: shape === 'ring' ? `2px solid ${color}` : 'none',
                          borderRadius: shape === 'circle' || shape === 'ring' ? '50%' : shape === 'diamond' ? 1 : 2,
                          transform: shape === 'diamond' ? 'rotate(45deg) scale(0.75)' : 'none',
                          boxShadow: glow ? `0 0 6px ${color}` : 'none',
                          opacity: 0.9,
                        }}
                      />
                      <span style={{ fontSize: 10, color: 'var(--text-secondary)' }}>{label}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Compact catalog counts */}
          {catalogCounts && (
            <div style={{ marginTop: 8, paddingTop: 6, borderTop: '1px solid var(--border)', fontSize: 9, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', lineHeight: 1.5 }}>
              {catalogCounts.fermi_all > 0 && <span>Fermi {formatCount(catalogCounts.fermi_all)} </span>}
              {catalogCounts.icecube > 0 && <span>| IceCube {formatCount(catalogCounts.icecube)} </span>}
              {catalogCounts.frb > 0 && <span>| FRB {catalogCounts.frb} </span>}
              {catalogCounts.pulsars > 0 && <span>| PSR {catalogCounts.pulsars}</span>}
            </div>
          )}
        </>
      )}
    </div>
  );
}
