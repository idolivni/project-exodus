/**
 * EXODUS Galaxy Explorer — Unexplainability Explanation Engine
 *
 * Mirrors the backend's _estimate_unexplainability logic to generate
 * human-readable explanations for WHY a target is unexplained (or explained).
 *
 * This is a frontend heuristic — the actual unexplainability score comes from
 * the pipeline. This just provides context/reasoning for display in tooltips.
 */

import type { ScoredTarget, ChannelDetail } from '../types';

export interface UnexExplanation {
  /** Overall verdict: 'unexplained' | 'partial' | 'explained' | 'unknown' */
  verdict: string;
  /** Color for the verdict */
  color: string;
  /** Short summary line */
  summary: string;
  /** Detailed explanation bullet points */
  reasons: string[];
  /** Suggested next steps */
  nextSteps: string[];
}

/** Channel display names */
const CH_NAMES: Record<string, string> = {
  ir_excess: 'IR Excess',
  ir_variability: 'IR Variability',
  proper_motion_anomaly: 'PM Anomaly',
  pm_anomaly: 'PM Anomaly',
  gaia_photometric: 'Gaia Photometric',
  gaia_photometric_anomaly: 'Gaia Photometric',
  transit_anomaly: 'Transit',
  radio_anomaly: 'Radio',
  radio_emission: 'Radio Emission',
  uv_anomaly: 'UV Anomaly',
  hr_anomaly: 'HR Anomaly',
  abundance_anomaly: 'Abundance',
  habitable_zone_planet: 'HZ Prior',
  hz_prior: 'HZ Prior',
};

/** Known binary template patterns */
const BINARY_PATTERNS = [
  { channels: ['proper_motion_anomaly', 'hr_anomaly'], label: 'PM + HR: classic unresolved binary signature' },
  { channels: ['proper_motion_anomaly', 'hr_anomaly', 'uv_anomaly'], label: 'PM + HR + UV: binary with chromospheric activity' },
  { channels: ['proper_motion_anomaly', 'gaia_photometric'], label: 'PM + Gaia Phot: astrometric binary' },
  { channels: ['ir_excess', 'proper_motion_anomaly'], label: 'IR + PM: possible disk + companion system' },
];

/** Get active channel keys from a target */
function getActiveChannels(target: ScoredTarget): string[] {
  if (!target.channel_details) return [];
  return Object.entries(target.channel_details)
    .filter(([, ch]) => ch.active)
    .map(([key]) => key);
}

/** Get active channel display names */
function getActiveChannelNames(target: ScoredTarget): string[] {
  return getActiveChannels(target).map(k => CH_NAMES[k] || k);
}

/** Check if channel pattern matches a known binary template */
function matchesBinaryTemplate(activeChannels: string[]): string | null {
  const activeSet = new Set(activeChannels);
  // Also check normalized keys
  const normalized = new Set(activeChannels.map(ch =>
    ch === 'pm_anomaly' ? 'proper_motion_anomaly' : ch
  ));

  for (const pattern of BINARY_PATTERNS) {
    if (pattern.channels.every(ch => normalized.has(ch) || activeSet.has(ch))) {
      return pattern.label;
    }
  }
  return null;
}

/**
 * Generate a human-readable explanation for a target's unexplainability score.
 */
export function explainUnexplainability(target: ScoredTarget): UnexExplanation {
  const unex = target.unexplainability_score;
  const risk = target.red_team_risk;
  const activeChannels = getActiveChannels(target);
  const activeNames = getActiveChannelNames(target);
  const nChannels = target.n_active_channels ?? activeChannels.length;

  // No data — can't explain
  if (unex == null) {
    return {
      verdict: 'unknown',
      color: '#888888',
      summary: 'No unexplainability score available',
      reasons: ['Target has not been scored for unexplainability yet.'],
      nextSteps: ['Run the full scoring pipeline to generate unexplainability estimates.'],
    };
  }

  const reasons: string[] = [];
  const nextSteps: string[] = [];

  // === EXPLAINED targets (unex < 0.2) ===
  if (unex < 0.2) {
    const binaryMatch = matchesBinaryTemplate(activeChannels);
    if (binaryMatch) {
      reasons.push(`Channel pattern matches binary template: ${binaryMatch}`);
    }

    if (risk) {
      const riskUpper = risk.toUpperCase();
      if (riskUpper === 'HIGH' || riskUpper === 'CRITICAL') {
        reasons.push(`Red-team risk: ${risk} \u2014 known astrophysical explanation likely`);
      } else {
        reasons.push(`Red-team risk: ${risk}`);
      }
    }

    // Check for specific channel combos that are commonly explained
    const hasIR = activeChannels.some(ch => ch.includes('ir'));
    const hasPM = activeChannels.some(ch => ch.includes('motion') || ch.includes('pm'));
    const hasHR = activeChannels.some(ch => ch.includes('hr'));
    const hasUV = activeChannels.some(ch => ch.includes('uv'));

    if (hasPM && hasHR && !hasIR) {
      reasons.push('PM + HR without IR excess: consistent with unresolved binary system');
    }
    if (hasPM && hasHR && hasUV && !hasIR) {
      reasons.push('UV excess in binary: chromospheric activity from tidally locked companion');
    }

    if (reasons.length === 0) {
      reasons.push('All detected anomalies have standard astrophysical explanations');
    }

    return {
      verdict: 'explained',
      color: '#44cc44',
      summary: `Explained (${unex.toFixed(3)}) \u2014 standard astrophysical origin`,
      reasons,
      nextSteps: ['No further follow-up needed for technosignature search.'],
    };
  }

  // === PARTIALLY EXPLAINED (0.2 <= unex < 0.5) ===
  if (unex < 0.5) {
    const binaryMatch = matchesBinaryTemplate(activeChannels);
    if (binaryMatch) {
      reasons.push(`Partial binary match: ${binaryMatch}, but residuals remain`);
    }

    if (nChannels >= 3) {
      reasons.push(`${nChannels}-channel convergence: some channels don't fit standard templates`);
    }

    // Check for interesting combos
    const hasRadio = activeChannels.some(ch => ch.includes('radio'));
    const hasIR = activeChannels.some(ch => ch.includes('ir'));

    if (hasRadio && hasIR) {
      reasons.push('Radio + IR co-detection: unusual for stellar sources');
    }

    if (risk) {
      reasons.push(`Red-team assessment: ${risk} risk`);
    }

    if (reasons.length === 0) {
      reasons.push('Some anomaly channels have partial explanations but residuals persist');
    }

    nextSteps.push('Deeper archival analysis recommended');
    if (nChannels >= 2) {
      nextSteps.push('Multi-wavelength follow-up could resolve ambiguity');
    }

    return {
      verdict: 'partial',
      color: '#ffaa00',
      summary: `Partially explained (${unex.toFixed(3)}) \u2014 some anomalies remain`,
      reasons,
      nextSteps,
    };
  }

  // === UNEXPLAINED (unex >= 0.5) ===
  if (nChannels >= 3) {
    reasons.push(`${nChannels}-channel convergence with NO standard template match`);
  }

  // Check for specific anomalous combos
  const hasRadio = activeChannels.some(ch => ch.includes('radio'));
  const hasIR = activeChannels.some(ch => ch.includes('ir'));
  const hasPM = activeChannels.some(ch => ch.includes('motion') || ch.includes('pm'));
  const hasUV = activeChannels.some(ch => ch.includes('uv'));

  if (hasRadio && hasIR) {
    reasons.push('Radio emission + IR excess: violates standard stellar templates');
    reasons.push('Radio luminosity may violate G\u00fcdel-Benz relation (X-ray/radio correlation)');
  }
  if (hasIR && hasPM && !hasRadio) {
    reasons.push('IR excess + PM anomaly without radio: disk + dynamical perturbation');
  }
  if (hasUV && hasIR) {
    reasons.push('Simultaneous UV + IR excess: unusual for single stellar model');
  }

  const binaryMatch = matchesBinaryTemplate(activeChannels);
  if (binaryMatch) {
    reasons.push(`Closest template: ${binaryMatch}, but template REJECTED by pipeline`);
  } else if (nChannels >= 2) {
    reasons.push('No known astrophysical template fits the multi-channel pattern');
  }

  if (risk) {
    const riskUpper = risk.toUpperCase();
    if (riskUpper === 'LOW') {
      reasons.push('Red-team: LOW risk \u2014 no obvious contamination or instrumental artifact');
    } else {
      reasons.push(`Red-team: ${risk} risk`);
    }
  }

  // Target-specific RUWE check
  if (target.channel_details) {
    const ruweDetail = target.channel_details['ruwe_anomaly'] || target.channel_details['gaia_photometric'];
    if (ruweDetail && !ruweDetail.active) {
      reasons.push('RUWE normal (single-star range) \u2014 argues AGAINST binary explanation');
    }
  }

  if (reasons.length === 0) {
    reasons.push('Multi-channel anomaly with no standard astrophysical explanation');
  }

  // Next steps for unexplained targets
  nextSteps.push('Priority target for follow-up observations');
  if (hasRadio) {
    nextSteps.push('VLA snapshot to confirm radio emission and measure spectral index');
  }
  if (hasIR) {
    nextSteps.push('Mid-IR spectroscopy (JWST/Spitzer) to characterize excess');
  }
  nextSteps.push('RV monitoring to test binary hypothesis (2+ epochs on 2m+ telescope)');

  return {
    verdict: 'unexplained',
    color: '#ff4444',
    summary: `Unexplained (${unex.toFixed(3)}) \u2014 no standard astrophysical origin identified`,
    reasons,
    nextSteps,
  };
}
