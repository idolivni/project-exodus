/**
 * EXODUS Galaxy Explorer — Story Generation Engine
 *
 * Phase 4: Auto-generates human-readable narratives for every target.
 * Stories explain what EXODUS found, what it means, and what's next.
 *
 * Template structure:
 * 1. Identity: What is this object?
 * 2. What EXODUS found: Which channels fired? How strongly?
 * 3. The explanation: What does the unexplainability scorer say?
 * 4. The skeptic's view: What does the Red-Team say?
 * 5. The verdict: Escalate, monitor, or demote?
 * 6. Context: Multi-messenger associations
 */

import type { ScoredTarget, ChannelDetail } from '../types';

// ── Types ────────────────────────────────────────────────

export interface TargetStory {
  /** One-line headline */
  headline: string;
  /** Full narrative (2-5 sentences) */
  narrative: string;
  /** Key finding bullet points */
  findings: string[];
  /** Suggested action */
  verdict: 'INVESTIGATE' | 'MONITOR' | 'EXPLAINED' | 'GATHERING';
  /** Color associated with verdict */
  verdictColor: string;
}

// ── Channel label mapping ────────────────────────────────

const CHANNEL_LABELS: Record<string, string> = {
  ir_excess: 'infrared excess',
  proper_motion_anomaly: 'proper motion anomaly',
  habitable_zone_planet: 'habitable zone planet',
  transit_anomaly: 'transit anomaly',
  radio_anomaly: 'radio anomaly',
  gaia_photometric_anomaly: 'photometric variability',
};

// ── Story generation ─────────────────────────────────────

export function generateStory(target: ScoredTarget): TargetStory {
  // Gathering state — data not yet scored
  if (target.status === 'gathered' || target.total_score === undefined) {
    return {
      headline: 'Data gathered, awaiting analysis',
      narrative: `${target.id} has been observed and data collected${target.distance_pc ? ` at ${target.distance_pc.toFixed(1)} pc` : ''}. ` +
        'The EXODUS pipeline is processing this target through its detection channels. ' +
        'Once scoring is complete, any anomalies will be reported here.',
      findings: [
        target.has_ir_data ? 'IR photometry collected' : null,
        target.has_gaia_data ? 'Gaia astrometry collected' : null,
        target.has_mm_data ? 'Multi-messenger crossmatches queued' : null,
      ].filter(Boolean) as string[],
      verdict: 'GATHERING',
      verdictColor: '#8888bb',
    };
  }

  const channels = target.channel_details || {};
  const activeChannels = Object.entries(channels).filter(([, ch]) => ch.active);
  const nActive = target.n_active_channels ?? activeChannels.length;
  const score = target.total_score;
  const unex = target.unexplainability_score;
  const risk = target.red_team_risk?.toUpperCase();

  // Build identity string
  const distStr = target.distance_pc ? `${target.distance_pc.toFixed(1)} pc away` : 'at unknown distance';
  const hostStr = target.host_star ? ` (host: ${target.host_star})` : '';

  // Build channel description
  const channelDescriptions = activeChannels.map(([key, ch]) => {
    const label = CHANNEL_LABELS[key] || key;
    return describeChannel(key, ch);
  });

  // Determine verdict
  let verdict: TargetStory['verdict'] = 'MONITOR';
  let verdictColor = '#ffaa00';

  if (unex != null && unex >= 0.5 && nActive >= 2) {
    verdict = 'INVESTIGATE';
    verdictColor = '#ff4444';
  } else if (unex != null && unex < 0.2) {
    verdict = 'EXPLAINED';
    verdictColor = '#44cc44';
  } else if (risk === 'CRITICAL' || risk === 'HIGH') {
    verdict = 'EXPLAINED';
    verdictColor = '#44cc44';
  }

  // Build headline
  let headline: string;
  if (verdict === 'INVESTIGATE') {
    headline = `${nActive}-channel convergence — unexplained anomaly`;
  } else if (verdict === 'EXPLAINED') {
    headline = nActive > 0
      ? `${nActive} channel${nActive > 1 ? 's' : ''} active — astrophysically explained`
      : 'No anomalies detected';
  } else {
    headline = nActive > 0
      ? `${nActive} channel${nActive > 1 ? 's' : ''} active — monitoring`
      : 'Baseline target — no anomalies';
  }

  // Build narrative
  const parts: string[] = [];

  // Identity
  parts.push(`${target.id}${hostStr} is ${distStr}.`);

  // What EXODUS found
  if (nActive === 0) {
    parts.push('EXODUS found no anomalous signals in any detection channel.');
  } else if (nActive === 1) {
    parts.push(`EXODUS detected a single anomaly: ${channelDescriptions[0]}.`);
  } else {
    parts.push(
      `EXODUS detected ${nActive} independent anomalies: ${channelDescriptions.join('; ')}.`
    );
  }

  // Score context
  if (score > 1.0) {
    parts.push(`The combined EXODUS score of ${score.toFixed(4)} places this target in the high-anomaly tier.`);
  } else if (score > 0.5) {
    parts.push(`The EXODUS score of ${score.toFixed(4)} indicates moderate anomaly significance.`);
  }

  // Unexplainability
  if (unex != null) {
    if (unex > 0.7) {
      parts.push(
        `The unexplainability score is ${unex.toFixed(3)} — no standard astrophysical template adequately explains the observed pattern.`
      );
    } else if (unex > 0.3) {
      parts.push(
        `The unexplainability score of ${unex.toFixed(3)} suggests partial explanation by known astrophysical models, but some residual signal remains.`
      );
    } else if (unex < 0.2) {
      parts.push(
        `The unexplainability score is only ${unex.toFixed(3)} — standard astrophysical templates fully explain the observed signatures.`
      );
    }
  }

  // Red-Team
  if (risk) {
    if (risk === 'LOW') {
      parts.push('The Red-Team engine found no concerning flags.');
    } else if (risk === 'MODERATE') {
      parts.push('The Red-Team engine flagged moderate risk factors that merit follow-up.');
    } else if (risk === 'HIGH' || risk === 'CRITICAL') {
      parts.push(
        `The Red-Team engine assigned ${risk} risk — likely explained by known astrophysical phenomena.`
      );
    }
  }

  // FDR
  if (target.fdr_significant) {
    parts.push('This target is FDR-significant after multiple-hypothesis correction.');
  }

  // Build findings
  const findings: string[] = [];

  for (const [key, ch] of activeChannels) {
    const label = CHANNEL_LABELS[key] || key;
    findings.push(`${label}: score ${ch.score.toFixed(3)}${ch.calibrated_p != null ? `, p=${ch.calibrated_p.toExponential(1)}` : ''}`);
  }

  if (target.stouffer_p != null) {
    findings.push(`Stouffer combined p-value: ${target.stouffer_p.toExponential(2)}`);
  }

  if (unex != null) {
    findings.push(`Unexplainability: ${unex.toFixed(3)}`);
  }

  if (target.hz_flag) {
    findings.push('Located in the habitable zone');
  }

  return {
    headline,
    narrative: parts.join(' '),
    findings,
    verdict,
    verdictColor,
  };
}

/** Generate a short channel description */
function describeChannel(key: string, ch: ChannelDetail): string {
  const score = ch.score.toFixed(3);
  const details = ch.details || {};

  switch (key) {
    case 'ir_excess':
      return `infrared excess (score ${score}${details.max_sigma ? `, ${(details.max_sigma as number).toFixed(1)}σ in ${details.max_band || 'WISE'}` : ''})`;
    case 'proper_motion_anomaly': {
      const parts: string[] = [];
      if (details.ruwe && (details.ruwe as number) > 1.4) {
        parts.push(`RUWE=${(details.ruwe as number).toFixed(2)}`);
      }
      if (details.wise_gaia_pm) {
        parts.push('Gaia-WISE PM discrepancy');
      }
      return `proper motion anomaly (score ${score}${parts.length ? `, ${parts.join(', ')}` : ''})`;
    }
    case 'habitable_zone_planet':
      return `habitable zone confirmation (score ${score})`;
    case 'transit_anomaly':
      return `transit anomaly (score ${score}${details.bls_power ? `, BLS power ${(details.bls_power as number).toFixed(3)}` : ''})`;
    case 'radio_anomaly':
      return `radio anomaly (score ${score})`;
    case 'gaia_photometric_anomaly':
      return `photometric variability (score ${score})`;
    default:
      return `${key} (score ${score})`;
  }
}
