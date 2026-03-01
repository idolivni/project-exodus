/**
 * EXODUS Galaxy Explorer — Type definitions
 */

// ── Target types ─────────────────────────────────────────

export interface Target {
  id: string;
  ra: number;
  dec: number;
  distance_pc?: number;
  hz_flag?: boolean;
  host_star?: string;
  status?: 'scored' | 'gathered';
}

export interface ScoredTarget extends Target {
  total_score: number;
  n_active_channels: number;
  stouffer_p?: number;
  fdr_significant?: boolean;
  q_value?: number;
  unexplainability_score?: number;
  red_team_risk?: string;
  channel_details?: Record<string, ChannelDetail>;
  // Gathered-only fields
  has_ir_data?: boolean;
  has_gaia_data?: boolean;
  has_mm_data?: boolean;
  _source?: 'report' | 'checkpoint';
}

export interface ChannelDetail {
  score: number;
  active: boolean;
  details?: Record<string, unknown>;
  calibrated_p?: number;
}

// ── Campaign types ───────────────────────────────────────

export interface Campaign {
  filename: string;
  campaign: string;
  description: string;
  n_targets: number;
  has_report?: boolean;
}

// ── Report types ─────────────────────────────────────────

export interface ReportInfo {
  filename: string;
  timestamp?: string;
  n_targets?: number;
  anomaly_count?: number;
  tier?: number;
  n_scored: number;
}

// ── Fermi types ──────────────────────────────────────────

export interface FermiSource {
  source_name?: string;
  ra: number;
  dec: number;
  class1?: string;
  signif_avg?: number;
}

// ── Multi-messenger types ────────────────────────────────

export interface IceCubeEvent {
  ra: number;
  dec: number;
  energy_gev: number;
  angular_err_deg?: number;
}

export interface FrbRepeater {
  name: string;
  ra: number;
  dec: number;
  dm?: number;
  n_bursts?: number;
}

export interface Pulsar {
  name: string;
  ra: number;
  dec: number;
  period_ms?: number;
  dm?: number;
}

// ── Report summary ───────────────────────────────────────

export interface ReportSummary {
  n_targets?: number;
  anomaly_count?: number;
  channels_active?: Record<string, number>;
  channels_calibrated?: string[];
  calibration_note?: string;
  multi_messenger?: Record<string, unknown>;
  top_targets: ScoredTarget[];
  red_team?: Record<string, unknown>;
  tier?: number;
  elapsed_min?: number;
  timestamp?: string;
}

// ── Layer system ─────────────────────────────────────────

export type ChannelId =
  | 'ir_excess'
  | 'ruwe_anomaly'
  | 'pm_discrepancy'
  | 'transit_anomaly'
  | 'gaia_phot'
  | 'radio'
  | 'hz_prior';

export type MmLayerId =
  | 'fermi_all'
  | 'fermi_unid'
  | 'icecube'
  | 'gw'
  | 'frb'
  | 'pulsar';

export interface LayerConfig {
  id: string;
  label: string;
  color: string;
  visible: boolean;
  opacity: number;
}

// ── Annotation types ─────────────────────────────────────

export interface Annotation {
  id: string;
  type: 'convergence_zone' | 'interesting' | 'false_positive' | 'environmental' | 'investigate';
  ra_center: number;
  dec_center: number;
  radius_deg?: number;
  notes?: string;
  created: string;
  targets_in_region?: string[];
}

// ── Pipeline status ──────────────────────────────────────

export interface PipelineStatus {
  running: boolean;
  message?: string;
  phase?: string;
  current_target?: string;
  progress?: number;
  total?: number;
  checkpoint_id?: string;
}

// ── Sky overlay (unified) ───────────────────────────────

export interface SkyOverlay {
  count: number;
  n_scored: number;
  n_gathered: number;
  targets: ScoredTarget[];
}

// ── Candidate types ─────────────────────────────────────

export interface TessResult {
  status: string;
  n_sectors: number;
  sectors: number[];
  baseline_days: number;
  rms_ppm: number;
  rms_pct: number;
  best_period_days: number;
  fap: number;
  verdict: string;
}

export interface Candidate {
  id: string;
  name: string;
  ra: number;
  dec: number;
  rank: number;
  teff_k: number;
  metallicity: number;
  gmag: number;
  distance_pc: number;
  spectral_type: string;
  exodus_score: number;
  n_channels: number;
  active_channels: string[];
  headline: string;
  highlights: string[];
  challenges: string[];
  next_steps: string[];
  tess?: TessResult;
  binary_dominance_context?: string;
  verdict: string;
  peer_review_confidence: number;
  peer_review_confirmed: number;
  peer_review_total: number;
  source_paper: string;
}
