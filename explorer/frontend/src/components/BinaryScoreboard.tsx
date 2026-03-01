/**
 * EXODUS Galaxy Explorer — Binary Dominance Scoreboard
 *
 * Compact, collapsible card showing campaign comparison data for
 * the binary dominance analysis. Data loaded dynamically from
 * the campaign comparison API endpoint.
 */

import { useState } from 'react';
import type { CampaignComparisonData } from '../hooks/useApi';

interface BinaryScoreboardProps {
  data: CampaignComparisonData | null;
}

export default function BinaryScoreboard({ data }: BinaryScoreboardProps) {
  const [isCollapsed, setIsCollapsed] = useState(true);

  if (!data) return null;

  const { campaigns, totals } = data;

  return (
    <div
      className="glass-panel"
      style={{
        position: 'absolute',
        bottom: 'calc(var(--statusbar-height, 32px) + 10px)',
        right: 12,
        width: 320,
        zIndex: 97,
        padding: isCollapsed ? '8px 12px' : '12px',
      }}
    >
      {/* Header — always visible */}
      <button
        onClick={() => setIsCollapsed(!isCollapsed)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          width: '100%',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          padding: 0,
          marginBottom: isCollapsed ? 0 : 10,
        }}
      >
        <span style={{ fontSize: 14 }}>{'\u{1F9EC}'}</span>
        <span
          style={{
            fontSize: 12,
            fontWeight: 600,
            color: 'var(--text-primary)',
            letterSpacing: '0.5px',
            flex: 1,
            textAlign: 'left',
          }}
        >
          Binary Dominance
        </span>
        <span
          style={{
            padding: '2px 8px',
            borderRadius: 10,
            background: 'rgba(255,215,0,0.15)',
            color: 'var(--gold)',
            fontSize: 11,
            fontWeight: 700,
            fontFamily: 'var(--font-mono)',
          }}
        >
          {totals.non_binary_3ch}/{totals.n}
        </span>
        <span style={{ color: 'var(--text-dim)', fontSize: 12, flexShrink: 0 }}>
          {isCollapsed ? '\u25B8' : '\u25BE'}
        </span>
      </button>

      {/* Expanded content */}
      {!isCollapsed && (
        <>
          {/* Table */}
          <table
            style={{
              width: '100%',
              borderCollapse: 'collapse',
              fontSize: 10,
              fontFamily: 'var(--font-mono)',
            }}
          >
            <thead>
              <tr
                style={{
                  borderBottom: '1px solid var(--border)',
                }}
              >
                <th
                  style={{
                    textAlign: 'left',
                    padding: '4px 6px',
                    fontSize: 9,
                    color: 'var(--text-dim)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                    fontWeight: 600,
                  }}
                >
                  Campaign
                </th>
                <th
                  style={{
                    textAlign: 'right',
                    padding: '4px 6px',
                    fontSize: 9,
                    color: 'var(--text-dim)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                    fontWeight: 600,
                  }}
                >
                  N
                </th>
                <th
                  style={{
                    textAlign: 'right',
                    padding: '4px 6px',
                    fontSize: 9,
                    color: 'var(--text-dim)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                    fontWeight: 600,
                  }}
                >
                  3ch
                </th>
                <th
                  style={{
                    textAlign: 'right',
                    padding: '4px 6px',
                    fontSize: 9,
                    color: 'var(--text-dim)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                    fontWeight: 600,
                  }}
                >
                  Non-binary
                </th>
              </tr>
            </thead>
            <tbody>
              {campaigns.map((row) => {
                const isHighlighted = row.non_binary_3ch > 0;
                return (
                  <tr
                    key={row.campaign}
                    style={{
                      background: isHighlighted ? 'rgba(255,215,0,0.08)' : 'transparent',
                      borderBottom: '1px solid rgba(255,255,255,0.04)',
                    }}
                  >
                    <td
                      style={{
                        padding: '4px 6px',
                        color: isHighlighted ? 'var(--gold)' : 'var(--text-secondary)',
                        fontWeight: isHighlighted ? 700 : 400,
                        maxWidth: 130,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                      title={row.campaign}
                    >
                      {row.campaign}
                    </td>
                    <td
                      style={{
                        padding: '4px 6px',
                        textAlign: 'right',
                        color: 'var(--text-secondary)',
                      }}
                    >
                      {row.n}
                    </td>
                    <td
                      style={{
                        padding: '4px 6px',
                        textAlign: 'right',
                        color: row.three_ch > 0 ? 'var(--text-primary)' : 'var(--text-dim)',
                      }}
                    >
                      {row.three_ch}
                    </td>
                    <td
                      style={{
                        padding: '4px 6px',
                        textAlign: 'right',
                        color: row.non_binary_3ch > 0 ? 'var(--gold)' : 'var(--text-dim)',
                        fontWeight: row.non_binary_3ch > 0 ? 700 : 400,
                      }}
                    >
                      {row.non_binary_3ch}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {/* Footer: totals + p-value */}
          <div
            style={{
              marginTop: 8,
              paddingTop: 8,
              borderTop: '1px solid var(--border)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <div style={{ fontSize: 10, color: 'var(--text-secondary)' }}>
              <span style={{ fontWeight: 700, color: 'var(--text-primary)' }}>
                {totals.non_binary_3ch}
              </span>
              <span style={{ color: 'var(--text-dim)' }}> / </span>
              <span>{totals.three_ch} 3ch</span>
              <span style={{ color: 'var(--text-dim)' }}> of </span>
              <span>{totals.n}</span>
            </div>
            <div
              style={{
                padding: '2px 8px',
                borderRadius: 4,
                fontSize: 9,
                fontWeight: 700,
                fontFamily: 'var(--font-mono)',
                background: 'rgba(255,215,0,0.1)',
                color: 'var(--gold)',
                border: '1px solid rgba(255,215,0,0.25)',
                whiteSpace: 'nowrap',
              }}
              title="Bonferroni-corrected p-value: clustering significance of non-binary 3ch targets (computed from campaign data)"
            >
              {data.totals && `${data.totals.non_binary_3ch} non-binary / ${data.totals.three_ch} 3ch`}
            </div>
          </div>

          {/* Summary line */}
          {data.summary && (
            <div
              style={{
                marginTop: 6,
                fontSize: 9,
                color: 'var(--text-dim)',
                lineHeight: 1.4,
                fontStyle: 'italic',
              }}
            >
              {data.summary}
            </div>
          )}
        </>
      )}
    </div>
  );
}
