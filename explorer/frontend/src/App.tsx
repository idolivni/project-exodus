/**
 * EXODUS Galaxy Explorer — Main App
 *
 * Phase 6: Full feature set — convergence, annotations, channel layers,
 * multi-messenger overlays, candidates, analysis workstation with
 * radar charts and side-by-side comparison.
 * Uses unified /api/sky/all endpoint for combined report + checkpoint data.
 */

import { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import './App.css';
import SkyMap from './components/SkyMap';
import ChannelBar from './components/ChannelBar';
import DetailPanel from './components/DetailPanel';
import ConvergencePanel from './components/ConvergencePanel';
import CandidatePanel from './components/CandidatePanel';
import AnnotationPanel from './components/AnnotationPanel';
import ComparePanel from './components/ComparePanel';
import BinaryScoreboard from './components/BinaryScoreboard';
import StatusBar from './components/StatusBar';
import {
  useSkyOverlay,
  useTargets,
  usePipelineStatus,
  useFermiSources,
  useIceCubeEvents,
  useFrbRepeaters,
  usePulsars,
  useCandidates,
  useAnnotations,
  useCampaignComparison,
} from './hooks/useApi';
import { computeConvergenceZones, DEFAULT_CONVERGENCE_CONFIG } from './utils/convergence';
import type { ConvergenceConfig } from './utils/convergence';
import type { ScoredTarget, LayerConfig } from './types';

const DEFAULT_LAYERS: LayerConfig[] = [
  // Detection channels
  { id: 'exodus_score', label: 'EXODUS Score', color: '#ffd700', visible: true, opacity: 1.0 },
  { id: 'ir_excess', label: 'IR Excess', color: '#ff9800', visible: false, opacity: 0.8 },
  { id: 'ruwe_anomaly', label: 'RUWE Anomaly', color: '#00bcd4', visible: false, opacity: 0.8 },
  { id: 'pm_discrepancy', label: 'PM Discrepancy', color: '#e040fb', visible: false, opacity: 0.8 },
  { id: 'hz_prior', label: 'HZ Prior', color: '#2196f3', visible: false, opacity: 0.8 },
  // Multi-messenger
  { id: 'fermi_unid', label: 'Fermi Unidentified', color: '#ffffff', visible: false, opacity: 0.6 },
  { id: 'fermi_all', label: 'Fermi All Sources', color: '#9e9e9e', visible: false, opacity: 0.4 },
  { id: 'icecube', label: 'IceCube (top 500)', color: '#9c27b0', visible: false, opacity: 0.5 },
  { id: 'frb', label: 'FRB Repeaters', color: '#f44336', visible: false, opacity: 0.8 },
  { id: 'pulsars', label: 'NANOGrav Pulsars', color: '#ffffff', visible: false, opacity: 0.8 },
];

export default function App() {
  // Data — unified sky overlay (reports + checkpoints)
  const { targets } = useTargets();
  const { overlay, loading: skyLoading, error: skyError } = useSkyOverlay();
  const pipelineStatus = usePipelineStatus();

  // Multi-messenger catalogs
  const { sources: fermiAll } = useFermiSources();
  const { sources: fermiUnid } = useFermiSources(true);
  const { events: icecubeEvents } = useIceCubeEvents(500);
  const { repeaters: frbRepeaters } = useFrbRepeaters();
  const { pulsars } = usePulsars();

  // Prime candidates
  const { candidates } = useCandidates();

  // Annotations (Phase 5)
  const { annotations, addAnnotation, deleteAnnotation } = useAnnotations();

  // Campaign comparison (Binary Dominance)
  const campaignComparison = useCampaignComparison();

  // UI state
  const [layers, setLayers] = useState<LayerConfig[]>(DEFAULT_LAYERS);
  const [selectedTarget, setSelectedTarget] = useState<ScoredTarget | null>(null);
  const [gotoTarget, setGotoTarget] = useState<ScoredTarget | null>(null);
  const [cursorRa, setCursorRa] = useState<number | null>(null);
  const [cursorDec, setCursorDec] = useState<number | null>(null);
  const [fov, setFov] = useState(180);

  // Distance filter
  const [distanceRange, setDistanceRange] = useState<[number, number]>([0, Infinity]);

  // Compare mode (Phase 6)
  const [compareA, setCompareA] = useState<ScoredTarget | null>(null);
  const [compareB, setCompareB] = useState<ScoredTarget | null>(null);
  const [compareMode, setCompareMode] = useState(false);

  // Convergence engine state
  const [convergenceConfig, setConvergenceConfig] = useState<ConvergenceConfig>(
    DEFAULT_CONVERGENCE_CONFIG
  );

  // Keyboard shortcut help overlay
  const [showHelp, setShowHelp] = useState(false);

  // Track current unexplained target index for N/P navigation
  const unexplainedIdxRef = useRef(0);

  // Apply distance filter to targets
  const filteredTargets = useMemo(() => {
    if (distanceRange[0] === 0 && distanceRange[1] === Infinity) return overlay.targets;
    return overlay.targets.filter(t => {
      if (t.distance_pc == null) return true; // show unknown-distance targets
      return t.distance_pc >= distanceRange[0] && t.distance_pc <= distanceRange[1];
    });
  }, [overlay.targets, distanceRange]);

  // Compute convergence zones (memoized — recomputes when data or config changes)
  const convergenceZones = useMemo(
    () =>
      computeConvergenceZones(
        filteredTargets,
        fermiUnid,
        icecubeEvents,
        frbRepeaters,
        pulsars,
        layers,
        convergenceConfig
      ),
    [filteredTargets, fermiUnid, icecubeEvents, frbRepeaters, pulsars, layers, convergenceConfig]
  );

  // Handlers
  const handleToggleLayer = useCallback((id: string) => {
    setLayers((prev) =>
      prev.map((l) => (l.id === id ? { ...l, visible: !l.visible } : l))
    );
  }, []);

  const handleOpacityChange = useCallback((id: string, opacity: number) => {
    setLayers((prev) =>
      prev.map((l) => (l.id === id ? { ...l, opacity } : l))
    );
  }, []);

  const handleTargetClick = useCallback((target: ScoredTarget) => {
    if (compareMode) {
      // In compare mode, fill the next empty slot
      if (!compareA) {
        setCompareA(target);
      } else if (!compareB) {
        setCompareB(target);
      } else {
        // Both filled — replace slot B
        setCompareB(target);
      }
    }
    setSelectedTarget(target);
  }, [compareMode, compareA, compareB]);

  const handlePositionChange = useCallback((ra: number, dec: number, newFov: number) => {
    setCursorRa(ra);
    setCursorDec(dec);
    setFov(newFov);
  }, []);

  const handleGotoTarget = useCallback((target: ScoredTarget) => {
    setGotoTarget(target);
    setSelectedTarget(target);
    // Also fill compare slot if in compare mode
    if (compareMode) {
      if (!compareA) setCompareA(target);
      else if (!compareB) setCompareB(target);
      else setCompareB(target);
    }
    setTimeout(() => {
      // Zoom to 2° FOV after navigation
      const al = (window as unknown as Record<string, unknown>).__aladin as { setFoV?: (f: number) => void } | undefined;
      if (al?.setFoV) al.setFoV(2);
      setGotoTarget(null);
    }, 150);
  }, [compareMode, compareA, compareB]);

  const handleCompareAdd = useCallback((target: ScoredTarget) => {
    if (!compareA) {
      setCompareA(target);
      setCompareMode(true);
    } else if (!compareB) {
      setCompareB(target);
    } else {
      // Both slots full — replace B
      setCompareB(target);
    }
    setCompareMode(true);
  }, [compareA, compareB]);

  const handleCompareClearSlot = useCallback((slot: 'A' | 'B') => {
    if (slot === 'A') setCompareA(null);
    else setCompareB(null);
  }, []);

  const handleCompareClose = useCallback(() => {
    setCompareA(null);
    setCompareB(null);
    setCompareMode(false);
  }, []);

  const handleGotoPosition = useCallback((ra: number, dec: number) => {
    // Create a temporary target to navigate to a position
    const tempTarget: ScoredTarget = {
      id: `annotation_${ra}_${dec}`,
      ra,
      dec,
      total_score: 0,
      n_active_channels: 0,
    };
    setGotoTarget(tempTarget);
    setTimeout(() => setGotoTarget(null), 100);
  }, []);

  // ── Global keyboard shortcuts ──────────────────────────────
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      switch (e.key) {
        case '?': { e.preventDefault(); setShowHelp(v => !v); break; }
        case 'Escape': {
          if (showHelp) setShowHelp(false);
          else if (compareMode) handleCompareClose();
          else if (selectedTarget) setSelectedTarget(null);
          break;
        }
        case 'n': case 'N': {
          e.preventDefault();
          const unexplained = convergenceZones
            .filter(z => z.unexplainability >= 0.5)
            .sort((a, b) => b.total_score - a.total_score);
          if (unexplained.length > 0) {
            if (e.key === 'N') {
              unexplainedIdxRef.current =
                (unexplainedIdxRef.current - 1 + unexplained.length) % unexplained.length;
            } else {
              unexplainedIdxRef.current =
                (unexplainedIdxRef.current + 1) % unexplained.length;
            }
            handleGotoTarget(unexplained[unexplainedIdxRef.current] as unknown as ScoredTarget);
          }
          break;
        }
        case 'c': {
          // Jump to first candidate from /api/candidates (if loaded)
          e.preventDefault();
          if (candidates.length > 0) {
            const c1 = filteredTargets.find(
              t => Math.abs(t.ra - candidates[0].ra) < 0.01 && Math.abs(t.dec - candidates[0].dec) < 0.01
            );
            if (c1) handleGotoTarget(c1);
          }
          break;
        }
        case 'C': {
          // Jump to second candidate from /api/candidates (if loaded)
          e.preventDefault();
          if (candidates.length > 1) {
            const c2 = filteredTargets.find(
              t => Math.abs(t.ra - candidates[1].ra) < 0.01 && Math.abs(t.dec - candidates[1].dec) < 0.01
            );
            if (c2) handleGotoTarget(c2);
          }
          break;
        }
        case '1': case '2': case '3': case '4': {
          e.preventDefault();
          const surveys: Record<string, string> = {
            '1': 'P/DSS2/color',
            '2': 'P/2MASS/color',
            '3': 'P/allWISE/color',
            '4': 'P/GALEXGR6_7/AIS/color',
          };
          const al = (window as unknown as Record<string, unknown>).__aladin as
            { setImageSurvey?: (s: string) => void } | undefined;
          if (al?.setImageSurvey) al.setImageSurvey(surveys[e.key]);
          break;
        }
        case 'h': {
          e.preventDefault();
          const al = (window as unknown as Record<string, unknown>).__aladin as
            { setFoV?: (f: number) => void } | undefined;
          if (al?.setFoV) al.setFoV(180);
          break;
        }
        case 'z': {
          e.preventDefault();
          const al = (window as unknown as Record<string, unknown>).__aladin as
            { setFoV?: (f: number) => void } | undefined;
          if (al?.setFoV) al.setFoV(2);
          break;
        }
        case 'Z': {
          e.preventDefault();
          const al = (window as unknown as Record<string, unknown>).__aladin as
            { setFoV?: (f: number) => void } | undefined;
          if (al?.setFoV) al.setFoV(0.2);
          break;
        }
        case 'g': {
          e.preventDefault();
          const al = (window as unknown as Record<string, unknown>).__aladin as
            { getFrame?: () => string; setFrame?: (f: string) => void } | undefined;
          if (al?.getFrame && al?.setFrame) {
            const cur = al.getFrame();
            al.setFrame(cur === 'GAL' ? 'ICRS' : 'GAL');
          }
          break;
        }
        case 's': { e.preventDefault(); handleToggleLayer('exodus_score'); break; }
        case 'i': { e.preventDefault(); handleToggleLayer('ir_excess'); break; }
        case 'r': { e.preventDefault(); handleToggleLayer('ruwe_anomaly'); break; }
        case 'p': { e.preventDefault(); handleToggleLayer('pm_discrepancy'); break; }
        default: break;
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [showHelp, compareMode, selectedTarget, convergenceZones, filteredTargets,
      handleGotoTarget, handleCompareClose, handleToggleLayer]);

  // Loading state
  if (skyLoading) {
    return (
      <div className="loading-overlay">
        <div className="loading-spinner" />
        <div className="loading-title">EXODUS</div>
        <div className="loading-subtitle">Loading galaxy data...</div>
      </div>
    );
  }

  return (
    <div className="app-container">
      {/* Error banner */}
      {skyError && (
        <div className="error-banner">
          API error: {skyError} — is the backend running? (uvicorn app:app --port 8000)
        </div>
      )}

      {/* Sky map */}
      <div className="sky-container">
        <SkyMap
          targets={filteredTargets}
          layers={layers}
          fermiAll={fermiAll}
          fermiUnid={fermiUnid}
          icecubeEvents={icecubeEvents}
          frbRepeaters={frbRepeaters}
          pulsars={pulsars}
          convergenceZones={convergenceZones}
          candidates={candidates}
          onTargetClick={handleTargetClick}
          onPositionChange={handlePositionChange}
          gotoTarget={gotoTarget}
        />
      </div>

      {/* Top bar — channel toggles, search, distance filter */}
      <ChannelBar
        layers={layers}
        onToggle={handleToggleLayer}
        onOpacityChange={handleOpacityChange}
        scoredCount={overlay.n_scored}
        gatheredCount={overlay.n_gathered}
        catalogCounts={{
          fermi_all: fermiAll.length,
          fermi_unid: fermiUnid.length,
          icecube: icecubeEvents.length,
          frb: frbRepeaters.length,
          pulsars: pulsars.length,
        }}
        onGotoTarget={handleGotoTarget}
        distanceRange={distanceRange}
        onDistanceRangeChange={setDistanceRange}
      />

      {/* Convergence panel (bottom-left, above status bar) */}
      <ConvergencePanel
        zones={convergenceZones}
        config={convergenceConfig}
        onConfigChange={setConvergenceConfig}
        onGotoZone={handleGotoTarget}
        computing={false}
      />

      {/* Annotation panel (bottom-left, next to convergence) */}
      <AnnotationPanel
        annotations={annotations}
        cursorRa={cursorRa}
        cursorDec={cursorDec}
        onAddAnnotation={addAnnotation}
        onDeleteAnnotation={deleteAnnotation}
        onGotoAnnotation={handleGotoPosition}
      />

      {/* Candidate panel (right side, always visible when candidates exist) */}
      {!selectedTarget && (
        <CandidatePanel
          candidates={candidates}
          onGotoCandidate={handleGotoTarget}
        />
      )}

      {/* Detail panel (right side, on target click — replaces candidate panel) */}
      <DetailPanel
        target={selectedTarget}
        onClose={() => setSelectedTarget(null)}
        candidates={candidates}
        onCompare={handleCompareAdd}
      />

      {/* Compare panel (bottom-right, Phase 6) */}
      {compareMode && (
        <ComparePanel
          targetA={compareA}
          targetB={compareB}
          onClose={handleCompareClose}
          onClearSlot={handleCompareClearSlot}
        />
      )}

      {/* Binary Dominance scoreboard (bottom-right, above status bar) */}
      <BinaryScoreboard data={campaignComparison} />

      {/* Status bar (bottom) */}
      <StatusBar
        cursorRa={cursorRa}
        cursorDec={cursorDec}
        fov={fov}
        pipelineStatus={pipelineStatus}
      />

      {/* Keyboard shortcut help overlay */}
      {showHelp && (
        <div
          style={{
            position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
            background: 'rgba(0,0,0,0.85)', zIndex: 9999,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}
          onClick={() => setShowHelp(false)}
        >
          <div
            style={{
              background: 'var(--bg-panel)', border: '1px solid var(--border)',
              borderRadius: 12, padding: '24px 32px', maxWidth: 520,
              color: 'var(--text-bright)', fontFamily: 'var(--font-mono)',
              fontSize: 12, lineHeight: 2.0,
            }}
            onClick={e => e.stopPropagation()}
          >
            <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 12, color: 'var(--gold)' }}>
              Keyboard Shortcuts
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '80px 1fr', gap: '2px 16px' }}>
              <span style={{ color: 'var(--gold)' }}>?</span><span>Toggle this help</span>
              <span style={{ color: 'var(--gold)' }}>Esc</span><span>Close panel / compare / help</span>
              <span style={{ color: 'var(--gold)' }}>n / N</span><span>Next / Previous unexplained target</span>
              <span style={{ color: 'var(--gold)' }}>c</span><span>Go to Candidate 1</span>
              <span style={{ color: 'var(--gold)' }}>C</span><span>Go to Candidate 2</span>
              <div style={{ gridColumn: '1/-1', height: 8 }} />
              <span style={{ color: 'var(--gold)' }}>1</span><span>Survey: DSS2 (optical)</span>
              <span style={{ color: 'var(--gold)' }}>2</span><span>Survey: 2MASS (near-IR)</span>
              <span style={{ color: 'var(--gold)' }}>3</span><span>Survey: AllWISE (mid-IR)</span>
              <span style={{ color: 'var(--gold)' }}>4</span><span>Survey: GALEX (UV)</span>
              <div style={{ gridColumn: '1/-1', height: 8 }} />
              <span style={{ color: 'var(--gold)' }}>h</span><span>Home (full sky, 180°)</span>
              <span style={{ color: 'var(--gold)' }}>z</span><span>Zoom to 2° (inspection)</span>
              <span style={{ color: 'var(--gold)' }}>Z</span><span>Deep zoom to 0.2° (close-up)</span>
              <span style={{ color: 'var(--gold)' }}>g</span><span>Toggle Galactic / ICRS frame</span>
              <div style={{ gridColumn: '1/-1', height: 8 }} />
              <span style={{ color: 'var(--gold)' }}>s</span><span>Toggle EXODUS Score layer</span>
              <span style={{ color: 'var(--gold)' }}>i</span><span>Toggle IR Excess layer</span>
              <span style={{ color: 'var(--gold)' }}>r</span><span>Toggle RUWE Anomaly layer</span>
              <span style={{ color: 'var(--gold)' }}>p</span><span>Toggle PM Discrepancy layer</span>
            </div>
            <div style={{ marginTop: 16, fontSize: 10, color: 'var(--text-dim)', textAlign: 'center' }}>
              Press ? or click outside to close
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
