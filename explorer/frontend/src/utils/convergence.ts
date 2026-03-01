/**
 * EXODUS Galaxy Explorer — Convergence Engine
 *
 * Phase 3: Spatial convergence detection across detection channels
 * and multi-messenger catalogs.
 *
 * For each scored target, counts how many distinct channels/catalogs
 * have a detection or source within a given angular radius.
 * Highlights "convergence zones" where multiple independent signals overlap.
 */

import type { ScoredTarget, FermiSource, IceCubeEvent, FrbRepeater, Pulsar, LayerConfig } from '../types';

// ── Types ────────────────────────────────────────────────

export interface ConvergenceZone {
  id: string;
  ra: number;
  dec: number;
  n_channels: number;
  channels: string[];
  /** The EXODUS target anchoring this zone (if any) */
  anchor_target?: ScoredTarget;
  /** Brief description */
  summary: string;
  /** Score of anchor target if available */
  score?: number;
  /** Unexplainability score from pipeline (0 = explained, 1 = completely unexplained) */
  unexplainability?: number;
  /** Red-team risk level from pipeline */
  red_team_risk?: string;
  /** Whether this zone is considered "explained" (low unexplainability or high risk = natural) */
  is_explained: boolean;
  /** FDR significance from anchor target */
  fdr_significant?: boolean;
}

export interface ConvergenceConfig {
  /** Angular match radius in degrees (used as base; per-source radii override) */
  radius_deg: number;
  /** Minimum channels to count as convergence */
  min_channels: number;
  /** Include multi-messenger catalogs in convergence count */
  include_mm: boolean;
}

export const DEFAULT_CONVERGENCE_CONFIG: ConvergenceConfig = {
  radius_deg: 0.5,
  min_channels: 2,
  include_mm: true,
};

/**
 * Optimal match radii per MM source type, derived from position uncertainty analysis.
 * SNR-optimized: balances detection probability vs chance coincidence rate.
 *   Fermi: sigma_pos ~0.2° → optimal r=0.3° (peak SNR=5.10)
 *   IceCube: sigma_pos ~1.0° for tracks → optimal r=1.5° (peak SNR=2.31)
 *   FRB: sigma_pos ~0.5° for CHIME → optimal r=0.8° (peak SNR=13.35)
 *   Pulsar: well-localized → r=0.1°
 */
const MM_OPTIMAL_RADII: Record<string, number> = {
  fermi: 0.3,
  icecube: 1.5,
  frb: 0.8,
  pulsar: 0.1,
};

// ── Angular distance ─────────────────────────────────────

/** Compute angular separation in degrees between two sky positions */
function angularSep(ra1: number, dec1: number, ra2: number, dec2: number): number {
  const deg2rad = Math.PI / 180;
  const d1 = dec1 * deg2rad;
  const d2 = dec2 * deg2rad;
  const dra = (ra2 - ra1) * deg2rad;
  const ddec = (dec2 - dec1) * deg2rad;
  // Haversine formula
  const a = Math.sin(ddec / 2) ** 2 + Math.cos(d1) * Math.cos(d2) * Math.sin(dra / 2) ** 2;
  return (2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))) / deg2rad;
}

// ── Channel key mapping ──────────────────────────────────
// Pipeline uses different keys than layer IDs

const CHANNEL_LAYER_MAP: Record<string, string> = {
  ir_excess: 'IR Excess',
  ir_variability: 'IR Variability',
  proper_motion_anomaly: 'PM Anomaly',
  pm_anomaly: 'PM Anomaly',
  habitable_zone_planet: 'HZ Prior',
  hz_prior: 'HZ Prior',
  transit_anomaly: 'Transit',
  radio_anomaly: 'Radio',
  radio_emission: 'Radio',
  uv_anomaly: 'UV Anomaly',
  hr_anomaly: 'HR Anomaly',
  abundance_anomaly: 'Abundance',
  gaia_photometric_anomaly: 'Gaia Phot',
  gaia_photometric: 'Gaia Phot',
};

// ── Main convergence computation ─────────────────────────

export function computeConvergenceZones(
  targets: ScoredTarget[],
  fermiUnid: FermiSource[],
  icecubeEvents: IceCubeEvent[],
  frbRepeaters: FrbRepeater[],
  pulsars: Pulsar[],
  layers: LayerConfig[],
  config: ConvergenceConfig
): ConvergenceZone[] {
  const { radius_deg, min_channels, include_mm } = config;
  const zones: ConvergenceZone[] = [];

  // Get visible layers for filtering
  const visibleLayerIds = new Set(layers.filter(l => l.visible).map(l => l.id));

  // For each scored target, check convergence
  const scoredTargets = targets.filter(
    t => t.total_score != null && t.channel_details
  );

  for (const target of scoredTargets) {
    const channelsHere: string[] = [];

    // 1. Count active EXODUS detection channels
    if (target.channel_details) {
      for (const [key, ch] of Object.entries(target.channel_details)) {
        if (!ch.active) continue;
        const label = CHANNEL_LAYER_MAP[key] || key;
        channelsHere.push(label);
      }
    }

    // 2. Count nearby multi-messenger sources (if enabled)
    //    Uses per-source optimal radii scaled by the user's radius slider.
    //    Slider at default (0.5°) → use optimal radii as-is.
    //    Slider higher/lower → scale all radii proportionally.
    if (include_mm) {
      const scale = radius_deg / DEFAULT_CONVERGENCE_CONFIG.radius_deg;

      // Fermi unidentified (optimal: 0.3°, position σ ~0.2°)
      const fermiRadius = MM_OPTIMAL_RADII.fermi * scale;
      const nearbyFermi = fermiUnid.some(
        s => angularSep(target.ra, target.dec, s.ra, s.dec) <= fermiRadius
      );
      if (nearbyFermi) channelsHere.push('Fermi Unid');

      // IceCube (optimal: 1.5°, position σ ~1.0° for tracks)
      const icecubeRadius = MM_OPTIMAL_RADII.icecube * scale;
      const nearbyIceCube = icecubeEvents.some(
        e => angularSep(target.ra, target.dec, e.ra, e.dec) <= icecubeRadius
      );
      if (nearbyIceCube) channelsHere.push('IceCube');

      // FRB (optimal: 0.8°, position σ ~0.5° for CHIME)
      const frbRadius = MM_OPTIMAL_RADII.frb * scale;
      const nearbyFrb = frbRepeaters.some(
        f => angularSep(target.ra, target.dec, f.ra, f.dec) <= frbRadius
      );
      if (nearbyFrb) channelsHere.push('FRB');

      // Pulsars (optimal: 0.1°, well-localized)
      const pulsarRadius = MM_OPTIMAL_RADII.pulsar * scale;
      const nearbyPulsar = pulsars.some(
        p => angularSep(target.ra, target.dec, p.ra, p.dec) <= pulsarRadius
      );
      if (nearbyPulsar) channelsHere.push('NANOGrav');
    }

    if (channelsHere.length >= min_channels) {
      const unex = target.unexplainability_score;
      const risk = target.red_team_risk;
      // A zone is "explained" if unexplainability is low (<0.2) OR red-team risk is high/critical
      const isExplained =
        (unex != null && unex < 0.2) ||
        (risk != null && ['HIGH', 'CRITICAL'].includes(risk.toUpperCase()));
      zones.push({
        id: target.id,
        ra: target.ra,
        dec: target.dec,
        n_channels: channelsHere.length,
        channels: channelsHere,
        anchor_target: target,
        summary: `${channelsHere.length} signals: ${channelsHere.join(' + ')}`,
        score: target.total_score,
        unexplainability: unex,
        red_team_risk: risk,
        is_explained: isExplained,
        fdr_significant: target.fdr_significant,
      });
    }
  }

  // Sort by channel count (descending), then by score
  zones.sort((a, b) => {
    if (b.n_channels !== a.n_channels) return b.n_channels - a.n_channels;
    return (b.score ?? 0) - (a.score ?? 0);
  });

  return zones;
}

/**
 * Assign a glow color based on convergence intensity.
 * More channels = brighter, more white-hot.
 */
export function convergenceColor(nChannels: number): string {
  if (nChannels >= 5) return 'rgba(255,255,255,0.9)';   // white-hot
  if (nChannels >= 4) return 'rgba(255,235,180,0.8)';   // warm white
  if (nChannels >= 3) return 'rgba(255,200,50,0.7)';    // bright gold
  return 'rgba(255,170,0,0.5)';                          // amber
}

/**
 * Convergence ring radius in pixels for Aladin overlay.
 * Scales with channel count for visual emphasis.
 */
export function convergenceSize(nChannels: number): number {
  if (nChannels >= 5) return 24;
  if (nChannels >= 4) return 20;
  if (nChannels >= 3) return 17;
  return 14;
}
