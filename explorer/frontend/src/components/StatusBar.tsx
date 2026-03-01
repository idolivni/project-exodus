/**
 * EXODUS Galaxy Explorer — Status Bar (bottom)
 *
 * Shows cursor position (RA/Dec), zoom level, pipeline status.
 * Pipeline status is checkpoint-aware — shows real-time progress.
 */

import { useState, useCallback } from 'react';
import type { PipelineStatus } from '../types';

interface StatusBarProps {
  cursorRa: number | null;
  cursorDec: number | null;
  fov: number;
  pipelineStatus: PipelineStatus;
}

function formatRA(ra: number): string {
  const h = Math.floor(ra / 15);
  const m = Math.floor((ra / 15 - h) * 60);
  const s = ((ra / 15 - h) * 60 - m) * 60;
  return `${h.toString().padStart(2, '0')}h ${m.toString().padStart(2, '0')}m ${s.toFixed(1).padStart(4, '0')}s`;
}

function formatDec(dec: number): string {
  const sign = dec >= 0 ? '+' : '-';
  const abs = Math.abs(dec);
  const d = Math.floor(abs);
  const m = Math.floor((abs - d) * 60);
  const s = ((abs - d) * 60 - m) * 60;
  return `${sign}${d.toString().padStart(2, '0')}\u00b0 ${m.toString().padStart(2, '0')}' ${s.toFixed(0).padStart(2, '0')}"`;
}

function formatFov(fov: number): string {
  if (fov >= 1) return `${fov.toFixed(1)}\u00b0`;
  if (fov >= 1 / 60) return `${(fov * 60).toFixed(1)}'`;
  return `${(fov * 3600).toFixed(0)}"`;
}

export default function StatusBar({ cursorRa, cursorDec, fov, pipelineStatus }: StatusBarProps) {
  const [copiedCursor, setCopiedCursor] = useState(false);
  const progress = pipelineStatus.progress;
  const total = pipelineStatus.total;
  const pct = progress && total ? ((progress / total) * 100).toFixed(0) : null;

  const handleCopyCursorCoords = useCallback(() => {
    if (cursorRa == null || cursorDec == null) return;
    const coordStr = `${cursorRa.toFixed(4)} ${cursorDec >= 0 ? '+' : ''}${cursorDec.toFixed(4)}`;
    navigator.clipboard.writeText(coordStr).then(() => {
      setCopiedCursor(true);
      setTimeout(() => setCopiedCursor(false), 1500);
    }).catch(() => {});
  }, [cursorRa, cursorDec]);

  return (
    <div
      style={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        height: 'var(--statusbar-height)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 16px',
        background: 'rgba(10,10,26,0.92)',
        borderTop: '1px solid var(--border)',
        zIndex: 100,
        fontFamily: 'var(--font-mono)',
        fontSize: 11,
      }}
    >
      {/* Left: cursor position (click to copy) */}
      <div
        onClick={handleCopyCursorCoords}
        title={cursorRa != null ? 'Click to copy coordinates' : undefined}
        style={{
          display: 'flex',
          gap: 16,
          color: copiedCursor ? 'var(--gold)' : 'var(--text-secondary)',
          cursor: cursorRa != null ? 'pointer' : 'default',
          transition: 'color 0.2s',
          userSelect: 'none',
        }}
      >
        {copiedCursor ? (
          <span style={{ color: 'var(--gold)' }}>Copied!</span>
        ) : cursorRa !== null && cursorDec !== null ? (
          <>
            <span>RA {formatRA(cursorRa)}</span>
            <span>Dec {formatDec(cursorDec)}</span>
            <span style={{ color: 'var(--text-dim)' }}>
              ({cursorRa.toFixed(4)}\u00b0, {cursorDec.toFixed(4)}\u00b0)
            </span>
          </>
        ) : (
          <span style={{ color: 'var(--text-dim)' }}>Hover over the sky...</span>
        )}
      </div>

      {/* Center: FOV */}
      <div style={{ color: 'var(--text-dim)' }}>FOV: {formatFov(fov)}</div>

      {/* Right: pipeline status */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        {pipelineStatus.running ? (
          <>
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: 'var(--success)',
                animation: 'pulse 1.5s infinite',
              }}
            />
            <span style={{ color: 'var(--success)' }}>
              {pipelineStatus.message || 'Pipeline running...'}
              {pct && (
                <span style={{ color: 'var(--text-dim)', marginLeft: 6 }}>({pct}%)</span>
              )}
            </span>
            {/* Mini progress bar */}
            {progress !== undefined && total !== undefined && total > 0 && (
              <div
                style={{
                  width: 60,
                  height: 3,
                  background: 'rgba(255,255,255,0.1)',
                  borderRadius: 2,
                  overflow: 'hidden',
                  marginLeft: 4,
                }}
              >
                <div
                  style={{
                    width: `${(progress / total) * 100}%`,
                    height: '100%',
                    background: 'var(--success)',
                    borderRadius: 2,
                    transition: 'width 0.5s ease',
                  }}
                />
              </div>
            )}
          </>
        ) : (
          <>
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: 'var(--text-dim)',
              }}
            />
            <span style={{ color: 'var(--text-dim)' }}>Pipeline idle</span>
          </>
        )}
      </div>
    </div>
  );
}
