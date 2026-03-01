/**
 * EXODUS Galaxy Explorer — Sky Map (Aladin Lite wrapper)
 *
 * Phase 3: Multi-messenger overlays + channel layers + convergence zones.
 * Uses Aladin Lite v3 for sky rendering + coordinate transforms.
 *
 * Features:
 * - Score-based marker coloring (red > orange > gold > blue)
 * - Size by active channel count
 * - Distinct gathered vs scored styling
 * - Multi-messenger catalog overlays (Fermi, IceCube, FRB, NANOGrav)
 * - Layer-aware visibility and opacity
 * - Convergence zone glow overlays
 * - Click-to-select targets
 */

import { useEffect, useRef, useCallback } from 'react';
import type { ScoredTarget, FermiSource, IceCubeEvent, FrbRepeater, Pulsar, LayerConfig, Candidate } from '../types';
import type { ConvergenceZone } from '../utils/convergence';
import { convergenceColor, convergenceSize } from '../utils/convergence';

// Aladin Lite is loaded as a global — we declare the type loosely
declare global {
  interface Window {
    A: {
      init: Promise<void>;
      aladin: (container: HTMLElement, options: Record<string, unknown>) => AladinInstance;
      catalog: (options: Record<string, unknown>) => AladinCatalog;
      source: (ra: number, dec: number, data: Record<string, unknown>) => AladinSource;
    };
  }
}

interface AladinInstance {
  setFov: (fov: number) => void;
  gotoRaDec: (ra: number, dec: number) => void;
  getRaDec: () => [number, number];
  getFov: () => [number, number];
  addCatalog: (cat: AladinCatalog) => void;
  removeLayers: () => void;
  on: (event: string, cb: (...args: unknown[]) => void) => void;
  setImageSurvey: (survey: string) => void;
}

interface AladinCatalog {
  addSources: (sources: AladinSource[]) => void;
  isShowing: boolean;
  hide: () => void;
  show: () => void;
  reportChange: () => void;
}

type AladinSource = unknown;

interface SkyMapProps {
  targets: ScoredTarget[];
  layers: LayerConfig[];
  fermiAll: FermiSource[];
  fermiUnid: FermiSource[];
  icecubeEvents: IceCubeEvent[];
  frbRepeaters: FrbRepeater[];
  pulsars: Pulsar[];
  convergenceZones: ConvergenceZone[];
  candidates: Candidate[];
  onTargetClick: (target: ScoredTarget) => void;
  onPositionChange: (ra: number, dec: number, fov: number) => void;
  gotoTarget?: ScoredTarget | null;
}

/** Channel color map — maps detection channel keys to their display colors */
const CHANNEL_COLORS: Record<string, string> = {
  ir_excess: '#ff9800',
  proper_motion_anomaly: '#e040fb',
  radio_emission: '#00e5ff',
  radio_anomaly: '#00e5ff',
  uv_anomaly: '#7c4dff',
  hr_anomaly: '#e91e63',
  gaia_photometric: '#00bcd4',
  habitable_zone_planet: '#2196f3',
  transit: '#4caf50',
  ir_variability: '#ff5722',
  abundance: '#8bc34a',
};

/** Channel display priority (rarest/most interesting first) */
const CHANNEL_PRIORITY = [
  'radio_emission', 'radio_anomaly', 'uv_anomaly', 'ir_excess',
  'proper_motion_anomaly', 'hr_anomaly', 'gaia_photometric',
  'habitable_zone_planet', 'transit', 'ir_variability', 'abundance',
];

/** Map target to marker color based on active channels */
function channelToColor(target: ScoredTarget): string {
  if (target.status === 'gathered' || !target.channel_details) return '#555588';

  const activeKeys = Object.entries(target.channel_details)
    .filter(([, ch]) => ch?.active)
    .map(([key]) => key);

  if (activeKeys.length === 0) return '#4488ff';
  if (activeKeys.length >= 3) return '#ffd700'; // gold for convergence

  // For 1-2 channels, use the highest-priority channel's color
  for (const p of CHANNEL_PRIORITY) {
    if (activeKeys.includes(p)) {
      return CHANNEL_COLORS[p] || '#ffaa00';
    }
  }
  return '#ffaa00';
}

/** Map target to marker shape based on channel count */
function channelToShape(target: ScoredTarget): string {
  if (target.status === 'gathered') return 'circle';
  const n = target.n_active_channels ?? 0;
  if (n >= 3) return 'diamond';
  if (n >= 2) return 'square';
  return 'circle';
}

/** Map active channel count to marker size */
function channelToSize(nChannels: number | undefined, status?: string): number {
  if (status === 'gathered') return 6;
  const n = nChannels ?? 0;
  if (n >= 3) return 16;
  if (n >= 2) return 12;
  if (n >= 1) return 9;
  return 7;
}

/** Get layer config by id */
function getLayer(layers: LayerConfig[], id: string): LayerConfig | undefined {
  return layers.find(l => l.id === id);
}

export default function SkyMap({
  targets,
  layers,
  fermiAll,
  fermiUnid,
  icecubeEvents,
  frbRepeaters,
  pulsars,
  convergenceZones,
  candidates,
  onTargetClick,
  onPositionChange,
  gotoTarget,
}: SkyMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const aladinRef = useRef<AladinInstance | null>(null);
  const ARef = useRef<typeof window.A | null>(null);

  // Track catalog refs for cleanup
  const catalogsRef = useRef<Map<string, AladinCatalog>>(new Map());

  // Initialize Aladin Lite
  useEffect(() => {
    if (!containerRef.current) return;

    const initAladin = async () => {
      const A = await import('aladin-lite');
      await (A as unknown as { default: { init: Promise<void> } }).default.init;

      const AApi = (A as unknown as { default: typeof window.A }).default;
      ARef.current = AApi;

      const aladin = AApi.aladin(containerRef.current!, {
        survey: 'P/2MASS/color',
        fov: 180,
        projection: 'AIT',
        cooFrame: 'J2000',
        showReticle: true,
        showZoomControl: true,
        showFullscreenControl: false,
        showLayersControl: true,
        showGotoControl: false,
        showSimbadPointerControl: false,
        showCooGridControl: false,
        reticleColor: 'rgba(255,215,0,0.6)',
        reticleSize: 22,
        backgroundColor: '#0a0a1a',
      });

      aladinRef.current = aladin;
      // Expose for debugging
      (window as unknown as Record<string, unknown>).__aladin = aladin;

      // Track position changes
      aladin.on('positionChanged', () => {
        const [ra, dec] = aladin.getRaDec();
        const [fovX] = aladin.getFov();
        onPositionChange(ra, dec, fovX);
      });

      // Object click
      aladin.on('objectClicked', (object: unknown) => {
        const obj = object as { data?: { _exodusTarget?: ScoredTarget } } | null;
        if (obj?.data?._exodusTarget) {
          onTargetClick(obj.data._exodusTarget);
        }
      });
    };

    initAladin();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Go to target when gotoTarget changes — pan only, no zoom
  // (DSS2 tiles can fail to load at smaller FOVs)
  useEffect(() => {
    if (gotoTarget && aladinRef.current) {
      aladinRef.current.gotoRaDec(gotoTarget.ra, gotoTarget.dec);
    }
  }, [gotoTarget]);

  // ── Helper: add or replace a named catalog ────────────────
  const addNamedCatalog = useCallback(
    (name: string, options: Record<string, unknown>, sources: AladinSource[]) => {
      const A = ARef.current;
      const aladin = aladinRef.current;
      if (!A || !aladin) return;

      // Create catalog
      const cat = A.catalog({ name, ...options });
      cat.addSources(sources);
      aladin.addCatalog(cat);
      catalogsRef.current.set(name, cat);
    },
    []
  );

  // ── Update all overlays when data or layers change ────────
  const updateOverlays = useCallback(() => {
    const A = ARef.current;
    const aladin = aladinRef.current;
    if (!A || !aladin) return;

    // Remove ALL previous catalog layers and rebuild
    aladin.removeLayers();
    catalogsRef.current.clear();

    const exodusLayer = getLayer(layers, 'exodus_score');
    const showExodus = exodusLayer?.visible ?? true;

    // ── EXODUS targets ──────────────────────────────────────
    if (showExodus) {
      const scored = targets.filter(t => t.status === 'scored' || t.total_score != null);
      const gathered = targets.filter(t => t.status === 'gathered' && t.total_score == null);

      // Gathered targets (grey dots)
      if (gathered.length > 0) {
        const gatheredSources = gathered.map((t) =>
          A.source(t.ra, t.dec, {
            name: t.id,
            status: 'Gathering data...',
            distance: t.distance_pc ? `${t.distance_pc.toFixed(1)} pc` : '?',
            _exodusTarget: t,
          })
        );
        addNamedCatalog('EXODUS Gathered', {
          shape: 'circle',
          sourceSize: 6,
          color: `rgba(85,85,136,${exodusLayer?.opacity ?? 1.0})`,
        }, gatheredSources);
      }

      // Scored targets (color-coded by score)
      if (scored.length > 0) {
        const scoredSources = scored.map((t) =>
          A.source(t.ra, t.dec, {
            name: t.id,
            score: t.total_score?.toFixed(4) ?? '?',
            channels: `${t.n_active_channels ?? 0} active`,
            distance: t.distance_pc ? `${t.distance_pc.toFixed(1)} pc` : '?',
            fdr: t.fdr_significant ? 'YES' : 'no',
            risk: t.red_team_risk ?? '?',
            _exodusTarget: t,
            _color: channelToColor(t),
            _shape: channelToShape(t),
            _size: channelToSize(t.n_active_channels, t.status),
          })
        );
        addNamedCatalog('EXODUS Scored', {
          shape: 'circle',
          sourceSize: 10,
          color: `rgba(255,215,0,${exodusLayer?.opacity ?? 1.0})`,
        }, scoredSources);
      }
    }

    // ── Knowledge State overlays ──────────────────────────────
    // Visual markers for high-interest targets (only when EXODUS score layer is on)
    if (showExodus) {
      const scoredTargets = targets.filter(t => t.total_score != null);

      // FDR Significant — bright white pulsing ring (rarest, most important)
      const fdrTargets = scoredTargets.filter(t => t.fdr_significant);
      if (fdrTargets.length > 0) {
        const fdrSources = fdrTargets.map((t) =>
          A.source(t.ra, t.dec, {
            name: `${t.id}`,
            type: 'FDR Significant',
            score: t.total_score?.toFixed(4) ?? '?',
            _exodusTarget: t,
          })
        );
        addNamedCatalog('FDR Significant', {
          shape: 'diamond',
          sourceSize: 16,
          color: 'rgba(255,255,255,0.9)',
        }, fdrSources);
        // Outer glow for FDR targets
        const fdrGlow = fdrTargets.map((t) =>
          A.source(t.ra, t.dec, { name: `${t.id} (fdr)` })
        );
        addNamedCatalog('FDR Glow', {
          shape: 'diamond',
          sourceSize: 24,
          color: 'rgba(255,255,255,0.25)',
        }, fdrGlow);
      }

      // INVESTIGATE — unexplainability > 0.5 and 2+ active channels
      const investigateTargets = scoredTargets.filter(t =>
        t.unexplainability_score != null &&
        t.unexplainability_score > 0.5 &&
        (t.n_active_channels ?? 0) >= 2 &&
        !t.fdr_significant  // don't double-mark FDR targets
      );
      if (investigateTargets.length > 0) {
        const invSources = investigateTargets.map((t) =>
          A.source(t.ra, t.dec, {
            name: `${t.id}`,
            type: 'INVESTIGATE',
            unexplainability: t.unexplainability_score?.toFixed(3) ?? '?',
            _exodusTarget: t,
          })
        );
        addNamedCatalog('Investigate', {
          shape: 'square',
          sourceSize: 16,
          color: 'rgba(255,68,68,0.85)',
        }, invSources);
      }

      // EXPLAINED — low unexplainability or high red-team risk
      const explainedTargets = scoredTargets.filter(t =>
        (t.unexplainability_score != null && t.unexplainability_score < 0.1) ||
        (t.red_team_risk != null && ['HIGH', 'CRITICAL'].includes(t.red_team_risk.toUpperCase()))
      );
      if (explainedTargets.length > 0) {
        const expSources = explainedTargets.map((t) =>
          A.source(t.ra, t.dec, {
            name: `${t.id}`,
            type: 'Explained',
            risk: t.red_team_risk ?? '?',
            _exodusTarget: t,
          })
        );
        addNamedCatalog('Explained', {
          shape: 'circle',
          sourceSize: 14,
          color: 'rgba(68,204,68,0.5)',
        }, expSources);
      }
    }

    // ── Fermi Unidentified ──────────────────────────────────
    const fermiUnidLayer = getLayer(layers, 'fermi_unid');
    if (fermiUnidLayer?.visible && fermiUnid.length > 0) {
      const sources = fermiUnid.map((s) =>
        A.source(s.ra, s.dec, {
          name: s.source_name ?? 'Fermi unID',
          type: 'Fermi 4FGL (unidentified)',
          class: s.class1 ?? 'unID',
          significance: s.signif_avg?.toFixed(1) ?? '?',
        })
      );
      addNamedCatalog('Fermi Unidentified', {
        shape: 'plus',
        sourceSize: 10,
        color: `rgba(255,255,255,${fermiUnidLayer.opacity})`,
      }, sources);
    }

    // ── Fermi All Sources ───────────────────────────────────
    const fermiAllLayer = getLayer(layers, 'fermi_all');
    if (fermiAllLayer?.visible && fermiAll.length > 0) {
      const sources = fermiAll.map((s) =>
        A.source(s.ra, s.dec, {
          name: s.source_name ?? 'Fermi',
          type: 'Fermi 4FGL',
          class: s.class1 ?? '?',
          significance: s.signif_avg?.toFixed(1) ?? '?',
        })
      );
      addNamedCatalog('Fermi All Sources', {
        shape: 'circle',
        sourceSize: 5,
        color: `rgba(158,158,158,${fermiAllLayer.opacity})`,
      }, sources);
    }

    // ── IceCube Neutrino Events ─────────────────────────────
    const icecubeLayer = getLayer(layers, 'icecube');
    if (icecubeLayer?.visible && icecubeEvents.length > 0) {
      const sources = icecubeEvents.map((e) =>
        A.source(e.ra, e.dec, {
          name: `IceCube ${e.energy_gev.toFixed(0)} GeV`,
          type: 'IceCube neutrino',
          energy: `${(e.energy_gev / 1e3).toFixed(1)} TeV`,
          error: e.angular_err_deg ? `±${e.angular_err_deg.toFixed(2)}°` : '?',
        })
      );
      addNamedCatalog('IceCube Events', {
        shape: 'diamond',
        sourceSize: 7,
        color: `rgba(156,39,176,${icecubeLayer.opacity})`,
      }, sources);
    }

    // ── FRB Repeaters ───────────────────────────────────────
    const frbLayer = getLayer(layers, 'frb');
    if (frbLayer?.visible && frbRepeaters.length > 0) {
      const sources = frbRepeaters.map((f) =>
        A.source(f.ra, f.dec, {
          name: f.name,
          type: 'FRB repeater',
          dm: f.dm ? `DM ${f.dm.toFixed(1)} pc/cm³` : '?',
          bursts: f.n_bursts ? `${f.n_bursts} bursts` : '?',
        })
      );
      addNamedCatalog('FRB Repeaters', {
        shape: 'cross',
        sourceSize: 12,
        color: `rgba(244,67,54,${frbLayer.opacity})`,
      }, sources);
    }

    // ── NANOGrav Pulsars ────────────────────────────────────
    const pulsarLayer = getLayer(layers, 'pulsars');
    if (pulsarLayer?.visible && pulsars.length > 0) {
      const sources = pulsars.map((p) =>
        A.source(p.ra, p.dec, {
          name: p.name,
          type: 'NANOGrav 15yr pulsar',
          period: p.period_ms ? `${p.period_ms.toFixed(3)} ms` : '?',
          dm: p.dm ? `DM ${p.dm.toFixed(2)}` : '?',
        })
      );
      addNamedCatalog('NANOGrav Pulsars', {
        shape: 'diamond',
        sourceSize: 10,
        color: `rgba(255,255,255,${pulsarLayer.opacity})`,
      }, sources);
    }

    // ── Channel-specific layers ─────────────────────────────
    // IR Excess: filter scored targets that have active ir_excess channel
    const irLayer = getLayer(layers, 'ir_excess');
    if (irLayer?.visible && targets.length > 0) {
      const irTargets = targets.filter(t =>
        t.channel_details?.ir_excess?.active
      );
      if (irTargets.length > 0) {
        const sources = irTargets.map((t) =>
          A.source(t.ra, t.dec, {
            name: t.id,
            type: 'IR Excess active',
            score: t.channel_details?.ir_excess?.score?.toFixed(4) ?? '?',
            p: t.channel_details?.ir_excess?.calibrated_p?.toFixed(4) ?? '?',
            _exodusTarget: t,
          })
        );
        addNamedCatalog('IR Excess', {
          shape: 'square',
          sourceSize: 10,
          color: `rgba(255,152,0,${irLayer.opacity})`,
        }, sources);
      }
    }

    // RUWE Anomaly — data key is 'proper_motion_anomaly', sub-detail 'ruwe'
    const ruweLayer = getLayer(layers, 'ruwe_anomaly');
    if (ruweLayer?.visible && targets.length > 0) {
      const ruweTargets = targets.filter(t => {
        const pm = t.channel_details?.proper_motion_anomaly;
        if (!pm?.active) return false;
        const ruwe = pm.details?.ruwe;
        return ruwe != null && ruwe > 1.4;
      });
      if (ruweTargets.length > 0) {
        const sources = ruweTargets.map((t) => {
          const pm = t.channel_details?.proper_motion_anomaly;
          return A.source(t.ra, t.dec, {
            name: t.id,
            type: 'RUWE Anomaly',
            ruwe: pm?.details?.ruwe?.toFixed(2) ?? '?',
            score: pm?.score?.toFixed(4) ?? '?',
            _exodusTarget: t,
          });
        });
        addNamedCatalog('RUWE Anomaly', {
          shape: 'square',
          sourceSize: 9,
          color: `rgba(0,188,212,${ruweLayer.opacity})`,
        }, sources);
      }
    }

    // PM Discrepancy — data key is 'proper_motion_anomaly', sub-detail 'wise_gaia_pm'
    const pmLayer = getLayer(layers, 'pm_discrepancy');
    if (pmLayer?.visible && targets.length > 0) {
      const pmTargets = targets.filter(t => {
        const pm = t.channel_details?.proper_motion_anomaly;
        return pm?.active && pm.details?.wise_gaia_pm;
      });
      if (pmTargets.length > 0) {
        const sources = pmTargets.map((t) => {
          const pm = t.channel_details?.proper_motion_anomaly;
          const wgpm = pm?.details?.wise_gaia_pm;
          return A.source(t.ra, t.dec, {
            name: t.id,
            type: 'PM Discrepancy',
            chi2: typeof wgpm === 'object' ? (wgpm as Record<string, unknown>)?.chi2?.toString() ?? '?' : '?',
            score: pm?.score?.toFixed(4) ?? '?',
            _exodusTarget: t,
          });
        });
        addNamedCatalog('PM Discrepancy', {
          shape: 'square',
          sourceSize: 9,
          color: `rgba(224,64,251,${pmLayer.opacity})`,
        }, sources);
      }
    }

    // HZ Prior — data key is 'habitable_zone_planet' in pipeline output
    const hzLayer = getLayer(layers, 'hz_prior');
    if (hzLayer?.visible && targets.length > 0) {
      const hzTargets = targets.filter(t =>
        t.channel_details?.habitable_zone_planet?.active
      );
      if (hzTargets.length > 0) {
        const sources = hzTargets.map((t) =>
          A.source(t.ra, t.dec, {
            name: t.id,
            type: 'Habitable Zone',
            score: t.channel_details?.habitable_zone_planet?.score?.toFixed(4) ?? '?',
            _exodusTarget: t,
          })
        );
        addNamedCatalog('HZ Prior', {
          shape: 'circle',
          sourceSize: 14,
          color: `rgba(33,150,243,${hzLayer.opacity})`,
        }, sources);
      }
    }
    // ── Convergence Zone overlays ──────────────────────────
    if (convergenceZones.length > 0) {
      const zoneSources = convergenceZones.map((z) =>
        A.source(z.ra, z.dec, {
          name: z.id,
          type: `Convergence: ${z.n_channels} signals`,
          channels: z.channels.join(', '),
          score: z.score?.toFixed(4) ?? '?',
          _exodusTarget: z.anchor_target ?? undefined,
        })
      );
      addNamedCatalog('Convergence Zones', {
        shape: 'circle',
        sourceSize: 20,
        color: 'rgba(255,200,50,0.7)',
      }, zoneSources);

      // Add a second layer of larger, fainter rings for glow effect
      const glowSources = convergenceZones
        .filter(z => z.n_channels >= 3)
        .map((z) =>
          A.source(z.ra, z.dec, {
            name: `${z.id} (glow)`,
            type: `${z.n_channels}-channel convergence`,
          })
        );
      if (glowSources.length > 0) {
        addNamedCatalog('Convergence Glow', {
          shape: 'circle',
          sourceSize: 30,
          color: 'rgba(255,220,100,0.3)',
        }, glowSources);
      }
    }

    // ── Prime Candidate highlights ────────────────────────
    // Large, bright markers that stand out from everything else
    if (candidates.length > 0) {
      // Outer glow ring (largest, faintest)
      const candidateGlow = candidates.map((c) =>
        A.source(c.ra, c.dec, {
          name: `${c.name} (glow)`,
          type: 'Prime Candidate',
        })
      );
      addNamedCatalog('Candidate Glow', {
        shape: 'circle',
        sourceSize: 40,
        color: 'rgba(255,215,0,0.2)',
      }, candidateGlow);

      // Middle ring
      const candidateMid = candidates.map((c) =>
        A.source(c.ra, c.dec, {
          name: `${c.name} (ring)`,
          type: 'Prime Candidate',
        })
      );
      addNamedCatalog('Candidate Ring', {
        shape: 'diamond',
        sourceSize: 24,
        color: 'rgba(255,215,0,0.6)',
      }, candidateMid);

      // Core marker (brightest, clickable)
      const candidateCore = candidates.map((c) => {
        // Find matching scored target for click handler
        const matchedTarget = targets.find(t => t.id === c.id);
        return A.source(c.ra, c.dec, {
          name: `${c.name}: ${c.headline}`,
          type: c.verdict === 'ELIMINATED' ? `ELIMINATED #${c.rank}` : `PRIME CANDIDATE #${c.rank}`,
          score: c.exodus_score.toFixed(2),
          channels: c.active_channels.join(', '),
          verdict: c.verdict,
          confidence: `${c.peer_review_confidence}%`,
          _exodusTarget: matchedTarget ?? {
            id: c.id,
            ra: c.ra,
            dec: c.dec,
            distance_pc: c.distance_pc,
            total_score: c.exodus_score,
            n_active_channels: c.n_channels,
            status: 'scored' as const,
          },
        });
      });
      addNamedCatalog('Prime Candidates', {
        shape: 'diamond',
        sourceSize: 16,
        color: 'rgba(255,255,255,0.95)',
      }, candidateCore);
    }
  }, [targets, layers, fermiAll, fermiUnid, icecubeEvents, frbRepeaters, pulsars, convergenceZones, candidates, addNamedCatalog]);

  // Debounced overlay update
  useEffect(() => {
    const timer = setTimeout(updateOverlays, 300);
    return () => clearTimeout(timer);
  }, [updateOverlays]);

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100%',
        position: 'absolute',
        top: 0,
        left: 0,
      }}
    />
  );
}
