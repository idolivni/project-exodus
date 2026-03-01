/**
 * EXODUS Galaxy Explorer — API hooks
 *
 * All data fetching from the FastAPI backend.
 * Uses the unified /api/sky/all endpoint for the main sky overlay.
 */

import { useState, useEffect, useCallback } from 'react';
import type { Target, ScoredTarget, Campaign, FermiSource, IceCubeEvent, FrbRepeater, Pulsar, ReportSummary, PipelineStatus, SkyOverlay, Candidate } from '../types';

const API_BASE = '/api';

async function fetchJson<T>(path: string): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`);
  if (!resp.ok) throw new Error(`API ${path}: ${resp.status}`);
  return resp.json();
}

// ── Targets ──────────────────────────────────────────────

export function useTargets(campaign = 'exodus_500') {
  const [targets, setTargets] = useState<Target[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    fetchJson<{ targets: Target[] }>(`/targets?campaign=${campaign}`)
      .then((data) => {
        setTargets(data.targets || []);
        setError(null);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [campaign]);

  return { targets, loading, error };
}

// ── Scored targets ───────────────────────────────────────

export function useScoredTargets() {
  const [scored, setScored] = useState<ScoredTarget[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    fetchJson<{ targets: ScoredTarget[] }>('/report/scored')
      .then((data) => {
        setScored(data.targets || []);
        setError(null);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return { scored, loading, error };
}

// ── Unified sky overlay (reports + checkpoints) ─────────

export function useSkyOverlay(pollInterval = 30000) {
  const [overlay, setOverlay] = useState<SkyOverlay>({ count: 0, n_scored: 0, n_gathered: 0, targets: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    fetchJson<SkyOverlay>('/sky/all')
      .then((data) => {
        setOverlay(data);
        setError(null);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    setLoading(true);
    refresh();
    const id = setInterval(refresh, pollInterval);
    return () => clearInterval(id);
  }, [refresh, pollInterval]);

  return { overlay, loading, error, refresh };
}

// ── Latest report ────────────────────────────────────────

export function useLatestReport() {
  const [report, setReport] = useState<ReportSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchJson<ReportSummary>('/report/latest')
      .then(setReport)
      .catch(() => setReport(null))
      .finally(() => setLoading(false));
  }, []);

  return { report, loading };
}

// ── Campaigns ────────────────────────────────────────────

export function useCampaigns() {
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);

  useEffect(() => {
    fetchJson<{ campaigns: Campaign[] }>('/campaigns')
      .then((data) => setCampaigns(data.campaigns || []))
      .catch(() => {});
  }, []);

  return campaigns;
}

// ── Fermi sources ────────────────────────────────────────

export function useFermiSources(unidOnly = false) {
  const [sources, setSources] = useState<FermiSource[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const qs = unidOnly ? '?unid_only=true' : '';
    fetchJson<{ sources: FermiSource[] }>(`/catalogs/fermi${qs}`)
      .then((data) => setSources(data.sources || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [unidOnly]);

  return { sources, loading };
}

// ── IceCube events ──────────────────────────────────────

export function useIceCubeEvents(topN = 500) {
  const [events, setEvents] = useState<IceCubeEvent[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchJson<{ events: IceCubeEvent[] }>(`/catalogs/icecube?top_n=${topN}`)
      .then((data) => setEvents(data.events || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [topN]);

  return { events, loading };
}

// ── FRB repeaters ───────────────────────────────────────

export function useFrbRepeaters() {
  const [repeaters, setRepeaters] = useState<FrbRepeater[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchJson<{ repeaters: FrbRepeater[] }>('/catalogs/frb')
      .then((data) => setRepeaters(data.repeaters || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return { repeaters, loading };
}

// ── NANOGrav pulsars ────────────────────────────────────

export function usePulsars() {
  const [pulsars, setPulsars] = useState<Pulsar[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchJson<{ pulsars: Pulsar[] }>('/catalogs/pulsars')
      .then((data) => setPulsars(data.pulsars || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return { pulsars, loading };
}

// ── Target detail ────────────────────────────────────────

export function useTargetDetail(targetId: string | null) {
  const [detail, setDetail] = useState<ScoredTarget | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback((id: string) => {
    setLoading(true);
    fetchJson<ScoredTarget>(`/target/${id}`)
      .then(setDetail)
      .catch(() => setDetail(null))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (targetId) load(targetId);
    else setDetail(null);
  }, [targetId, load]);

  return { detail, loading };
}

// ── Pipeline status ──────────────────────────────────────

export function usePipelineStatus(pollInterval = 10000) {
  const [status, setStatus] = useState<PipelineStatus>({ running: false });

  useEffect(() => {
    const poll = () => {
      fetchJson<PipelineStatus>('/pipeline/status')
        .then(setStatus)
        .catch(() => {});
    };
    poll();
    const id = setInterval(poll, pollInterval);
    return () => clearInterval(id);
  }, [pollInterval]);

  return status;
}

// ── Candidates ──────────────────────────────────────────

export function useCandidates() {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchJson<{ candidates: Candidate[] }>('/candidates')
      .then((data) => setCandidates(data.candidates || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return { candidates, loading };
}

// ── Search ───────────────────────────────────────────────

export function useSearch() {
  const [results, setResults] = useState<ScoredTarget[]>([]);
  const [loading, setLoading] = useState(false);

  const search = useCallback((query: string) => {
    if (!query || query.length < 2) {
      setResults([]);
      return;
    }
    setLoading(true);
    fetchJson<{ results: ScoredTarget[] }>(`/search/${encodeURIComponent(query)}`)
      .then((data) => setResults(data.results || []))
      .catch(() => setResults([]))
      .finally(() => setLoading(false));
  }, []);

  return { results, loading, search };
}

// ── Annotations ─────────────────────────────────────────

export function useAnnotations() {
  const [annotations, setAnnotations] = useState<import('../types').Annotation[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(() => {
    fetchJson<{ annotations: import('../types').Annotation[] }>('/annotations')
      .then((data) => setAnnotations(data.annotations || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const addAnnotation = useCallback(async (ann: {
    type: string;
    ra_center: number;
    dec_center: number;
    radius_deg?: number;
    notes?: string;
    targets_in_region?: string[];
  }) => {
    try {
      const resp = await fetch(`${API_BASE}/annotations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(ann),
      });
      if (resp.ok) {
        refresh();
        return true;
      }
    } catch {}
    return false;
  }, [refresh]);

  const deleteAnnotation = useCallback(async (id: string) => {
    try {
      const resp = await fetch(`${API_BASE}/annotations/${id}`, { method: 'DELETE' });
      if (resp.ok) {
        refresh();
        return true;
      }
    } catch {}
    return false;
  }, [refresh]);

  return { annotations, loading, addAnnotation, deleteAnnotation, refresh };
}

// ── Campaign Comparison (Binary Dominance) ──────────────

export interface CampaignComparisonData {
  campaigns: { campaign: string; n: number; three_ch: number; non_binary_3ch: number; pattern: string }[];
  totals: { n: number; three_ch: number; non_binary_3ch: number };
  summary: string;
}

export function useCampaignComparison() {
  const [data, setData] = useState<CampaignComparisonData | null>(null);

  useEffect(() => {
    fetchJson<CampaignComparisonData>('/campaign_comparison')
      .then(setData)
      .catch(() => console.warn('Campaign comparison endpoint not available'));
  }, []);

  return data;
}
