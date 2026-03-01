/**
 * EXODUS Galaxy Explorer — Target Detail Panel (slide-in from right)
 *
 * Phase 4: Full EXODUS score breakdown + auto-generated story narrative.
 * Handles both scored and gathered (pending) targets.
 *
 * Features:
 * - Score badge with channel count, FDR, Stouffer
 * - Auto-generated narrative story
 * - Channel breakdown with per-channel scores
 * - Risk badges (Red-Team, Unexplainability, HZ)
 * - External links (SIMBAD, Aladin)
 * - Knowledge state indicator
 */

import { useState, useMemo, useCallback } from 'react';
import type { ScoredTarget, Candidate } from '../types';
import { generateStory } from '../utils/story';
import { explainUnexplainability } from '../utils/explainUnex';
import ChannelRadar from './ChannelRadar';

interface DetailPanelProps {
  target: ScoredTarget | null;
  onClose: () => void;
  candidates?: Candidate[];
  onCompare?: (target: ScoredTarget) => void;
}

const CHANNEL_NAMES: Record<string, { label: string; color: string }> = {
  ir_excess: { label: 'IR Excess', color: 'var(--ch-ir)' },
  ir_variability: { label: 'IR Variability', color: '#ff6b35' },
  proper_motion_anomaly: { label: 'PM Anomaly', color: 'var(--ch-pm)' },
  pm_anomaly: { label: 'PM Anomaly', color: 'var(--ch-pm)' },
  gaia_photometric: { label: 'Gaia Phot', color: 'var(--ch-phot)' },
  gaia_photometric_anomaly: { label: 'Gaia Phot', color: 'var(--ch-phot)' },
  transit_anomaly: { label: 'Transit', color: 'var(--ch-transit)' },
  radio: { label: 'Radio', color: 'var(--ch-radio)' },
  radio_anomaly: { label: 'Radio Anomaly', color: 'var(--ch-radio)' },
  radio_emission: { label: 'Radio Emission', color: '#00e5ff' },
  uv_anomaly: { label: 'UV Anomaly', color: '#7c4dff' },
  hr_anomaly: { label: 'HR Anomaly', color: '#e91e63' },
  abundance_anomaly: { label: 'Abundance', color: '#4caf50' },
  hz_prior: { label: 'HZ Prior', color: 'var(--ch-hz)' },
  habitable_zone_planet: { label: 'HZ Prior', color: 'var(--ch-hz)' },
};

function riskColor(risk?: string): string {
  if (!risk) return 'var(--text-dim)';
  switch (risk.toUpperCase()) {
    case 'CRITICAL':
      return '#ff2222';
    case 'HIGH':
      return '#ff4444';
    case 'MODERATE':
      return '#ff8800';
    case 'LOW':
      return '#44cc44';
    default:
      return 'var(--text-dim)';
  }
}

/** Knowledge state progression */
function getKnowledgeState(target: ScoredTarget): {
  label: string;
  color: string;
  progress: number;
} {
  if (target.status === 'gathered') {
    return { label: 'DETECTED', color: '#8888bb', progress: 25 };
  }
  const nActive = target.n_active_channels ?? 0;
  const hasUnex = target.unexplainability_score != null;
  const hasRedTeam = target.red_team_risk != null;

  if (hasRedTeam && hasUnex) {
    if (target.unexplainability_score! > 0.5) {
      return { label: 'UNEXPLAINED', color: '#ff4444', progress: 80 };
    }
    return { label: 'EXPLAINED', color: '#44cc44', progress: 100 };
  }
  if (nActive > 0) {
    return { label: 'CHARACTERIZED', color: '#ffaa00', progress: 60 };
  }
  return { label: 'SCANNED', color: '#6688aa', progress: 40 };
}

type TabId = 'story' | 'data' | 'links';

export default function DetailPanel({ target, onClose, candidates, onCompare }: DetailPanelProps) {
  const [activeTab, setActiveTab] = useState<TabId>('story');
  const [showUnexTooltip, setShowUnexTooltip] = useState(false);
  const [copiedCoords, setCopiedCoords] = useState(false);

  const story = useMemo(
    () => (target ? generateStory(target) : null),
    [target]
  );

  // Generate unexplainability explanation
  const unexExplanation = useMemo(
    () => (target ? explainUnexplainability(target) : null),
    [target]
  );

  // Check if this target is a prime candidate
  const candidate = useMemo(
    () => candidates?.find(c => c.id === target?.id) ?? null,
    [candidates, target]
  );

  // Click-to-copy coordinates
  const handleCopyCoords = useCallback(() => {
    if (!target) return;
    const coordStr = `${target.ra?.toFixed(4)} ${target.dec && target.dec >= 0 ? '+' : ''}${target.dec?.toFixed(4)}`;
    navigator.clipboard.writeText(coordStr).then(() => {
      setCopiedCoords(true);
      setTimeout(() => setCopiedCoords(false), 1500);
    }).catch(() => {});
  }, [target]);

  if (!target) return null;

  const isGathered = target.status === 'gathered';
  const channels = target.channel_details || {};
  const knowledgeState = getKnowledgeState(target);

  return (
    <div
      className="glass-panel"
      style={{
        position: 'absolute',
        top: 'calc(var(--topbar-height, 44px) + 10px)',
        right: 12,
        width: 'var(--detail-width)',
        maxHeight: 'calc(100vh - var(--topbar-height, 44px) - var(--statusbar-height, 32px) - 30px)',
        overflow: 'auto',
        zIndex: 100,
        padding: '16px',
        animation: 'slideInRight 0.2s ease-out',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: 8,
        }}
      >
        <div>
          <h3
            style={{
              fontSize: 15,
              fontWeight: 600,
              color: 'var(--text-bright)',
              fontFamily: 'var(--font-mono)',
            }}
          >
            {target.id}
          </h3>
          <div
            className="mono"
            onClick={handleCopyCoords}
            title="Click to copy coordinates"
            style={{
              color: copiedCoords ? 'var(--gold)' : 'var(--text-secondary)',
              fontSize: 11,
              marginTop: 2,
              cursor: 'pointer',
              transition: 'color 0.2s',
            }}
          >
            {copiedCoords ? (
              <span style={{ color: 'var(--gold)' }}>Copied!</span>
            ) : (
              <>
                RA {target.ra?.toFixed(4)}&deg; &middot; Dec {target.dec?.toFixed(4)}&deg;
                {target.distance_pc && ` \u00b7 ${target.distance_pc.toFixed(1)} pc`}
                <span style={{ fontSize: 9, marginLeft: 4, opacity: 0.5 }}>📋</span>
              </>
            )}
          </div>
          {target.host_star && (
            <div style={{ color: 'var(--text-dim)', fontSize: 10, marginTop: 2 }}>
              {target.host_star}
            </div>
          )}
        </div>
        <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
          {onCompare && !isGathered && (
            <button
              onClick={() => onCompare(target)}
              title="Add to comparison"
              style={{
                padding: '3px 8px',
                background: 'rgba(255,215,0,0.08)',
                border: '1px solid rgba(255,215,0,0.2)',
                borderRadius: 4,
                color: 'var(--gold)',
                cursor: 'pointer',
                fontSize: 9,
                fontWeight: 600,
                letterSpacing: '0.3px',
                transition: 'all 0.15s',
              }}
            >
              ⚖️ Compare
            </button>
          )}
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-dim)',
              cursor: 'pointer',
              fontSize: 18,
              padding: '0 4px',
              lineHeight: 1,
            }}
          >
            &times;
          </button>
        </div>
      </div>

      {/* Candidate badge */}
      {candidate && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '8px 12px',
            marginBottom: 10,
            background: candidate.verdict === 'ELIMINATED'
              ? 'linear-gradient(135deg, rgba(255,68,68,0.12), rgba(204,34,34,0.08))'
              : 'linear-gradient(135deg, rgba(255,215,0,0.12), rgba(255,152,0,0.08))',
            borderRadius: 6,
            border: candidate.verdict === 'ELIMINATED'
              ? '1px solid rgba(255,68,68,0.25)'
              : '1px solid rgba(255,215,0,0.25)',
          }}
        >
          <span style={{ fontSize: 14 }}>{candidate.verdict === 'ELIMINATED' ? '❌' : '🎯'}</span>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: candidate.verdict === 'ELIMINATED' ? '#ff4444' : '#ffd700' }}>
              {candidate.verdict === 'ELIMINATED' ? 'ELIMINATED' : 'PRIME CANDIDATE'} #{candidate.rank}
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginTop: 1 }}>
              {candidate.headline}
            </div>
          </div>
          <div
            style={{
              padding: '2px 8px',
              borderRadius: 4,
              fontSize: 9,
              fontWeight: 600,
              background: 'rgba(68,204,68,0.15)',
              color: '#44cc44',
              border: '1px solid rgba(68,204,68,0.3)',
            }}
          >
            {candidate.peer_review_confirmed}/{candidate.peer_review_total} PR
          </div>
        </div>
      )}

      {/* Knowledge state bar */}
      <div style={{ marginBottom: 12 }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 4,
          }}
        >
          <span
            style={{
              fontSize: 9,
              fontWeight: 600,
              color: knowledgeState.color,
              textTransform: 'uppercase',
              letterSpacing: '1px',
            }}
          >
            {knowledgeState.label}
          </span>
          <span style={{ fontSize: 9, color: 'var(--text-dim)' }}>
            Knowledge State
          </span>
        </div>
        <div
          style={{
            width: '100%',
            height: 3,
            background: 'rgba(255,255,255,0.06)',
            borderRadius: 2,
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              width: `${knowledgeState.progress}%`,
              height: '100%',
              background: knowledgeState.color,
              borderRadius: 2,
              transition: 'width 0.3s ease',
            }}
          />
        </div>
      </div>

      {/* Tab bar */}
      <div
        style={{
          display: 'flex',
          gap: 2,
          marginBottom: 12,
          borderBottom: '1px solid var(--border)',
          paddingBottom: 1,
        }}
      >
        {(['story', 'data', 'links'] as TabId[]).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              padding: '4px 12px',
              fontSize: 11,
              fontWeight: activeTab === tab ? 600 : 400,
              color: activeTab === tab ? 'var(--text-bright)' : 'var(--text-dim)',
              background: activeTab === tab ? 'rgba(255,255,255,0.06)' : 'transparent',
              border: 'none',
              borderBottom: activeTab === tab ? '2px solid var(--gold)' : '2px solid transparent',
              borderRadius: '4px 4px 0 0',
              cursor: 'pointer',
              textTransform: 'capitalize',
              transition: 'all 0.15s',
            }}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* ══ STORY TAB ══════════════════════════════════════════ */}
      {activeTab === 'story' && story && (
        <div>
          {/* Verdict badge */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              marginBottom: 12,
              padding: '8px 12px',
              background: `${story.verdictColor}11`,
              borderRadius: 6,
              border: `1px solid ${story.verdictColor}33`,
            }}
          >
            <span
              style={{
                width: 10,
                height: 10,
                borderRadius: '50%',
                background: story.verdictColor,
                flexShrink: 0,
                boxShadow: story.verdict === 'INVESTIGATE' ? `0 0 8px ${story.verdictColor}` : 'none',
              }}
            />
            <div>
              <div
                style={{
                  fontSize: 12,
                  fontWeight: 600,
                  color: story.verdictColor,
                }}
              >
                {story.headline}
              </div>
            </div>
          </div>

          {/* Narrative */}
          <div
            style={{
              fontSize: 12,
              lineHeight: 1.6,
              color: 'var(--text-primary)',
              marginBottom: 14,
              padding: '0 2px',
            }}
          >
            {story.narrative}
          </div>

          {/* Key findings */}
          {story.findings.length > 0 && (
            <div style={{ marginBottom: 14 }}>
              <div
                style={{
                  fontSize: 10,
                  color: 'var(--text-dim)',
                  textTransform: 'uppercase',
                  letterSpacing: '1px',
                  marginBottom: 6,
                }}
              >
                Key Findings
              </div>
              {story.findings.map((f, i) => (
                <div
                  key={i}
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 6,
                    fontSize: 11,
                    color: 'var(--text-secondary)',
                    marginBottom: 3,
                    fontFamily: 'var(--font-mono)',
                  }}
                >
                  <span style={{ color: 'var(--text-dim)', flexShrink: 0 }}>•</span>
                  <span>{f}</span>
                </div>
              ))}
            </div>
          )}

          {/* Score summary (compact) */}
          {!isGathered && (
            <div
              style={{
                display: 'flex',
                gap: 12,
                padding: '8px 10px',
                background: 'rgba(255,255,255,0.03)',
                borderRadius: 6,
                fontFamily: 'var(--font-mono)',
                fontSize: 11,
              }}
            >
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: scoreColor(target.total_score) }}>
                  {target.total_score?.toFixed(4) ?? '\u2014'}
                </div>
                <div style={{ fontSize: 9, color: 'var(--text-dim)', textTransform: 'uppercase' }}>
                  Score
                </div>
              </div>
              <div style={{ borderLeft: '1px solid var(--border)', paddingLeft: 12, display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 1 }}>
                <div style={{ fontSize: 10 }}>
                  <span style={{ color: 'var(--text-dim)' }}>Channels: </span>
                  <span style={{ color: 'var(--text-bright)' }}>{target.n_active_channels ?? 0}</span>
                </div>
                <div style={{ fontSize: 10 }}>
                  <span style={{ color: 'var(--text-dim)' }}>FDR: </span>
                  <span style={{ color: target.fdr_significant ? 'var(--success)' : 'var(--text-dim)' }}>
                    {target.fdr_significant ? 'YES' : 'no'}
                  </span>
                </div>
                <div style={{ fontSize: 10 }}>
                  <span style={{ color: 'var(--text-dim)' }}>Stouffer: </span>
                  <span>{target.stouffer_p?.toExponential(2) ?? '\u2014'}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ══ DATA TAB ═══════════════════════════════════════════ */}
      {activeTab === 'data' && (
        <div>
          {/* Gathered-only status */}
          {isGathered && (
            <div
              style={{
                padding: '10px 12px',
                background: 'rgba(85,85,136,0.15)',
                borderRadius: 6,
                marginBottom: 16,
                border: '1px solid rgba(85,85,136,0.3)',
              }}
            >
              <div style={{ fontSize: 12, color: '#8888bb', fontWeight: 500 }}>
                Data Gathered — Awaiting Scoring
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 4 }}>
                {target.has_ir_data && <span style={{ marginRight: 8 }}>IR data</span>}
                {target.has_gaia_data && <span style={{ marginRight: 8 }}>Gaia data</span>}
                {target.has_mm_data && <span>MM matches</span>}
              </div>
            </div>
          )}

          {/* Badges row */}
          {!isGathered && (
            <div
              style={{ display: 'flex', gap: 6, marginBottom: 14, flexWrap: 'wrap' }}
            >
              {target.red_team_risk && (
                <span
                  style={{
                    padding: '2px 8px',
                    borderRadius: 4,
                    fontSize: 10,
                    fontWeight: 600,
                    background: riskColor(target.red_team_risk) + '22',
                    color: riskColor(target.red_team_risk),
                    border: `1px solid ${riskColor(target.red_team_risk)}44`,
                    textTransform: 'uppercase',
                  }}
                >
                  {target.red_team_risk} risk
                </span>
              )}
              {target.unexplainability_score != null && (
                <div style={{ position: 'relative', display: 'inline-block' }}>
                  <span
                    onMouseEnter={() => setShowUnexTooltip(true)}
                    onMouseLeave={() => setShowUnexTooltip(false)}
                    style={{
                      padding: '2px 8px',
                      borderRadius: 4,
                      fontSize: 10,
                      fontWeight: 600,
                      background:
                        target.unexplainability_score > 0.5
                          ? 'rgba(255,215,0,0.15)'
                          : 'rgba(255,255,255,0.05)',
                      color:
                        target.unexplainability_score > 0.5
                          ? 'var(--gold)'
                          : 'var(--text-dim)',
                      border: '1px solid rgba(255,255,255,0.1)',
                      cursor: 'help',
                    }}
                  >
                    Unex: {target.unexplainability_score.toFixed(3)} {target.unexplainability_score > 0.5 ? '⚠' : '✓'}
                  </span>
                  {/* Why Unexplained? Tooltip */}
                  {showUnexTooltip && unexExplanation && (
                    <div
                      style={{
                        position: 'absolute',
                        top: '100%',
                        left: 0,
                        marginTop: 6,
                        width: 280,
                        padding: '10px 12px',
                        background: 'var(--bg-card)',
                        border: '1px solid var(--border-active)',
                        borderRadius: 8,
                        boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
                        zIndex: 500,
                        animation: 'fadeIn 0.15s ease-out',
                      }}
                    >
                      {/* Verdict header */}
                      <div style={{
                        fontSize: 11,
                        fontWeight: 700,
                        color: unexExplanation.color,
                        marginBottom: 6,
                        textTransform: 'uppercase',
                        letterSpacing: '0.3px',
                      }}>
                        {unexExplanation.verdict === 'unexplained' ? '⚠ Why Unexplained?' : unexExplanation.verdict === 'explained' ? '✓ Why Explained' : 'Analysis'}
                      </div>
                      <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginBottom: 8, lineHeight: 1.4 }}>
                        {unexExplanation.summary}
                      </div>

                      {/* Reasons */}
                      <div style={{ marginBottom: 6 }}>
                        {unexExplanation.reasons.map((r, i) => (
                          <div key={i} style={{
                            display: 'flex', gap: 5, fontSize: 9,
                            color: 'var(--text-secondary)', marginBottom: 3,
                            lineHeight: 1.4, fontFamily: 'var(--font-mono)',
                          }}>
                            <span style={{ color: unexExplanation.color, flexShrink: 0 }}>•</span>
                            <span>{r}</span>
                          </div>
                        ))}
                      </div>

                      {/* Next steps */}
                      {unexExplanation.nextSteps.length > 0 && (
                        <>
                          <div style={{
                            fontSize: 9, color: 'var(--text-dim)', textTransform: 'uppercase',
                            letterSpacing: '0.5px', marginBottom: 3, marginTop: 4,
                            borderTop: '1px solid var(--border)', paddingTop: 6,
                          }}>
                            Next Steps
                          </div>
                          {unexExplanation.nextSteps.map((s, i) => (
                            <div key={i} style={{
                              display: 'flex', gap: 5, fontSize: 9,
                              color: 'var(--text-dim)', marginBottom: 2,
                              lineHeight: 1.4,
                            }}>
                              <span style={{ flexShrink: 0 }}>→</span>
                              <span>{s}</span>
                            </div>
                          ))}
                        </>
                      )}
                    </div>
                  )}
                </div>
              )}
              {target.hz_flag && (
                <span
                  style={{
                    padding: '2px 8px',
                    borderRadius: 4,
                    fontSize: 10,
                    fontWeight: 600,
                    background: 'rgba(33,150,243,0.15)',
                    color: 'var(--ch-hz)',
                    border: '1px solid rgba(33,150,243,0.3)',
                  }}
                >
                  HZ
                </span>
              )}
            </div>
          )}

          {/* Channel radar chart */}
          {!isGathered && Object.keys(channels).length >= 3 && (
            <div
              style={{
                display: 'flex',
                justifyContent: 'center',
                padding: '4px 0 8px',
                marginBottom: 8,
                borderBottom: '1px solid var(--border)',
              }}
            >
              <ChannelRadar channels={channels} size={140} showLabels={true} />
            </div>
          )}

          {/* Channel breakdown */}
          {!isGathered && Object.keys(channels).length > 0 && (
            <>
              <div
                style={{
                  fontSize: 11,
                  color: 'var(--text-dim)',
                  textTransform: 'uppercase',
                  marginBottom: 8,
                  letterSpacing: '1px',
                }}
              >
                Channel Scores
              </div>

              {Object.entries(channels).map(([key, ch]) => {
                const info = CHANNEL_NAMES[key] || { label: key, color: '#888' };
                return (
                  <div
                    key={key}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      padding: '5px 8px',
                      marginBottom: 2,
                      borderRadius: 4,
                      background: ch.active
                        ? 'rgba(255,255,255,0.04)'
                        : 'transparent',
                    }}
                  >
                    <span
                      style={{
                        width: 8,
                        height: 8,
                        borderRadius: '50%',
                        background: ch.active ? info.color : 'transparent',
                        border: `2px solid ${info.color}`,
                        opacity: ch.active ? 1 : 0.3,
                        flexShrink: 0,
                      }}
                    />
                    <span
                      style={{
                        flex: 1,
                        fontSize: 12,
                        color: ch.active
                          ? 'var(--text-primary)'
                          : 'var(--text-dim)',
                      }}
                    >
                      {info.label}
                    </span>
                    <span
                      className="mono"
                      style={{
                        color: ch.active
                          ? 'var(--text-bright)'
                          : 'var(--text-dim)',
                        fontSize: 11,
                      }}
                    >
                      {ch.score?.toFixed(3) ?? '\u2014'}
                    </span>
                    {ch.calibrated_p != null && (
                      <span
                        className="mono"
                        style={{ color: 'var(--text-dim)', fontSize: 10 }}
                      >
                        p={ch.calibrated_p.toExponential(1)}
                      </span>
                    )}
                  </div>
                );
              })}

              {/* Q-value row */}
              {target.q_value != null && (
                <div
                  style={{
                    marginTop: 8,
                    padding: '4px 8px',
                    fontSize: 11,
                    color: 'var(--text-dim)',
                    fontFamily: 'var(--font-mono)',
                  }}
                >
                  q-value (FDR): {target.q_value.toExponential(2)}
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* ══ LINKS TAB ══════════════════════════════════════════ */}
      {activeTab === 'links' && (
        <div>
          <div
            style={{
              fontSize: 10,
              color: 'var(--text-dim)',
              textTransform: 'uppercase',
              letterSpacing: '1px',
              marginBottom: 10,
            }}
          >
            External Databases
          </div>

          {[
            {
              label: 'SIMBAD',
              desc: 'Object identification',
              url: `https://simbad.u-strasbg.fr/simbad/sim-coo?Coord=${target.ra}+${target.dec}&CooFrame=FK5&CooEpoch=2000&CooEqui=2000&Radius=10&Radius.unit=arcsec`,
            },
            {
              label: 'Aladin Lite',
              desc: 'Interactive sky view',
              url: `https://aladin.cds.unistra.fr/AladinLite/?target=${target.ra}+${target.dec}&fov=0.1&survey=P/DSS2/color`,
            },
            {
              label: 'NExScI',
              desc: 'NASA Exoplanet Archive',
              url: `https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=pscomppars&where=ra%20between%20${(target.ra - 0.01).toFixed(4)}%20and%20${(target.ra + 0.01).toFixed(4)}%20and%20dec%20between%20${(target.dec - 0.01).toFixed(4)}%20and%20${(target.dec + 0.01).toFixed(4)}`,
            },
            {
              label: 'VizieR',
              desc: 'Catalog query',
              url: `https://vizier.cds.unistra.fr/viz-bin/VizieR-4?-c=${target.ra}+${target.dec}&-c.rs=10&-out.max=20`,
            },
            {
              label: 'Gaia DR3',
              desc: 'ESA archive',
              url: `https://gea.esac.esa.int/archive/?#query=SELECT%20*%20FROM%20gaiadr3.gaia_source%20WHERE%20CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',${target.ra},${target.dec},0.003))=1`,
            },
          ].map(({ label, desc, url }) => (
            <a
              key={label}
              href={url}
              target="_blank"
              rel="noreferrer"
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '8px 10px',
                marginBottom: 4,
                borderRadius: 6,
                background: 'rgba(255,255,255,0.03)',
                color: 'var(--text-primary)',
                textDecoration: 'none',
                border: '1px solid transparent',
                transition: 'all 0.15s',
                fontSize: 12,
              }}
            >
              <div>
                <div style={{ fontWeight: 500 }}>{label}</div>
                <div style={{ fontSize: 10, color: 'var(--text-dim)', marginTop: 1 }}>{desc}</div>
              </div>
              <span style={{ color: 'var(--text-dim)', fontSize: 14 }}>↗</span>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

function scoreColor(score?: number): string {
  if (!score) return 'var(--text-dim)';
  if (score >= 0.5) return '#ff4444';
  if (score >= 0.2) return '#ff9800';
  return '#ffd700';
}
