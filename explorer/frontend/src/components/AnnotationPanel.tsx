/**
 * EXODUS Galaxy Explorer — Annotation Panel
 *
 * Phase 5: Mark regions on the sky, persist annotations, type classification.
 *
 * Features:
 * - Create annotations at current cursor position
 * - Type classification: convergence_zone, interesting, false_positive, environmental, investigate
 * - Notes field for free-text descriptions
 * - Annotation list with click-to-navigate
 * - Delete annotations
 * - Persistent storage via backend
 */

import { useState, useCallback } from 'react';
import type { Annotation } from '../types';

interface AnnotationPanelProps {
  annotations: Annotation[];
  cursorRa: number | null;
  cursorDec: number | null;
  onAddAnnotation: (ann: {
    type: string;
    ra_center: number;
    dec_center: number;
    radius_deg?: number;
    notes?: string;
  }) => Promise<boolean>;
  onDeleteAnnotation: (id: string) => Promise<boolean>;
  onGotoAnnotation: (ra: number, dec: number) => void;
}

const ANNOTATION_TYPES: { value: string; label: string; color: string; icon: string }[] = [
  { value: 'interesting', label: 'Interesting', color: '#ffd700', icon: '★' },
  { value: 'investigate', label: 'Investigate', color: '#ff4444', icon: '🔍' },
  { value: 'convergence_zone', label: 'Convergence', color: '#ffaa00', icon: '🎯' },
  { value: 'false_positive', label: 'False Positive', color: '#888888', icon: '✗' },
  { value: 'environmental', label: 'Environmental', color: '#44cc44', icon: '🌍' },
];

function getTypeInfo(type: string) {
  return ANNOTATION_TYPES.find(t => t.value === type) || ANNOTATION_TYPES[0];
}

export default function AnnotationPanel({
  annotations,
  cursorRa,
  cursorDec,
  onAddAnnotation,
  onDeleteAnnotation,
  onGotoAnnotation,
}: AnnotationPanelProps) {
  const [collapsed, setCollapsed] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [formType, setFormType] = useState('interesting');
  const [formNotes, setFormNotes] = useState('');
  const [formRadius, setFormRadius] = useState(0.1);
  const [saving, setSaving] = useState(false);

  const handleCreate = useCallback(async () => {
    if (cursorRa == null || cursorDec == null) return;
    setSaving(true);
    const ok = await onAddAnnotation({
      type: formType,
      ra_center: cursorRa,
      dec_center: cursorDec,
      radius_deg: formRadius,
      notes: formNotes,
    });
    setSaving(false);
    if (ok) {
      setFormNotes('');
      setShowForm(false);
    }
  }, [cursorRa, cursorDec, formType, formRadius, formNotes, onAddAnnotation]);

  return (
    <div
      className="glass-panel"
      style={{
        position: 'absolute',
        bottom: 'calc(var(--statusbar-height, 32px) + 40px)',
        left: 12,
        width: collapsed ? 'auto' : 240,
        maxHeight: collapsed ? 'auto' : 360,
        overflow: collapsed ? 'hidden' : 'auto',
        zIndex: 98,
        padding: collapsed ? '6px 10px' : '12px',
        transition: 'all 0.2s ease',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          cursor: 'pointer',
          marginBottom: collapsed ? 0 : 10,
        }}
        onClick={() => setCollapsed(!collapsed)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 13 }}>📌</span>
          <span
            style={{
              fontSize: 13,
              fontWeight: 600,
              color: 'var(--text-bright)',
              letterSpacing: '0.3px',
            }}
          >
            Annotations
          </span>
          {annotations.length > 0 && (
            <span
              className="mono"
              style={{
                fontSize: 10,
                padding: '1px 6px',
                borderRadius: 8,
                background: 'rgba(255,215,0,0.2)',
                color: '#ffd700',
                fontWeight: 600,
              }}
            >
              {annotations.length}
            </span>
          )}
        </div>
        <span style={{ color: 'var(--text-dim)', fontSize: 12 }}>
          {collapsed ? '▸' : '▾'}
        </span>
      </div>

      {!collapsed && (
        <>
          {/* Create new annotation button */}
          {!showForm ? (
            <button
              onClick={() => setShowForm(true)}
              disabled={cursorRa == null}
              style={{
                width: '100%',
                padding: '6px 10px',
                marginBottom: 10,
                background: cursorRa != null ? 'rgba(255,215,0,0.1)' : 'rgba(255,255,255,0.03)',
                border: '1px dashed rgba(255,215,0,0.3)',
                borderRadius: 6,
                color: cursorRa != null ? '#ffd700' : 'var(--text-dim)',
                fontSize: 11,
                cursor: cursorRa != null ? 'pointer' : 'default',
                transition: 'all 0.15s',
              }}
            >
              + Mark current position
              {cursorRa != null && (
                <span className="mono" style={{ fontSize: 9, display: 'block', color: 'var(--text-dim)', marginTop: 2 }}>
                  RA {cursorRa.toFixed(4)}° Dec {cursorDec?.toFixed(4)}°
                </span>
              )}
            </button>
          ) : (
            /* Annotation creation form */
            <div
              style={{
                padding: '10px',
                marginBottom: 10,
                background: 'rgba(255,215,0,0.05)',
                border: '1px solid rgba(255,215,0,0.2)',
                borderRadius: 6,
              }}
            >
              <div style={{ fontSize: 10, color: 'var(--text-dim)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                New Annotation
              </div>

              {/* Position */}
              <div className="mono" style={{ fontSize: 9, color: 'var(--text-secondary)', marginBottom: 8 }}>
                RA {cursorRa?.toFixed(4)}° Dec {cursorDec?.toFixed(4)}°
              </div>

              {/* Type selector */}
              <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap', marginBottom: 8 }}>
                {ANNOTATION_TYPES.map(t => (
                  <button
                    key={t.value}
                    onClick={() => setFormType(t.value)}
                    style={{
                      padding: '3px 7px',
                      borderRadius: 4,
                      fontSize: 9,
                      fontWeight: formType === t.value ? 700 : 400,
                      background: formType === t.value ? `${t.color}22` : 'rgba(255,255,255,0.03)',
                      color: formType === t.value ? t.color : 'var(--text-dim)',
                      border: formType === t.value ? `1px solid ${t.color}44` : '1px solid transparent',
                      cursor: 'pointer',
                    }}
                  >
                    {t.icon} {t.label}
                  </button>
                ))}
              </div>

              {/* Radius */}
              <div style={{ marginBottom: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-dim)', marginBottom: 2 }}>
                  <span>Radius</span>
                  <span className="mono">{formRadius >= 1 ? `${formRadius.toFixed(1)}°` : `${(formRadius * 60).toFixed(0)}'`}</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={60}
                  value={formRadius * 60}
                  onChange={e => setFormRadius(Number(e.target.value) / 60)}
                  style={{ width: '100%', height: 2, accentColor: '#ffd700', cursor: 'pointer' }}
                />
              </div>

              {/* Notes */}
              <textarea
                placeholder="Notes (optional)..."
                value={formNotes}
                onChange={e => setFormNotes(e.target.value)}
                style={{
                  width: '100%',
                  padding: '5px 8px',
                  background: 'rgba(255,255,255,0.05)',
                  border: '1px solid var(--border)',
                  borderRadius: 4,
                  color: 'var(--text-primary)',
                  fontSize: 10,
                  fontFamily: 'var(--font-mono)',
                  resize: 'vertical',
                  minHeight: 40,
                  maxHeight: 80,
                  outline: 'none',
                  marginBottom: 8,
                  boxSizing: 'border-box',
                }}
              />

              {/* Actions */}
              <div style={{ display: 'flex', gap: 6, justifyContent: 'flex-end' }}>
                <button
                  onClick={() => setShowForm(false)}
                  style={{
                    padding: '4px 10px',
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid var(--border)',
                    borderRadius: 4,
                    color: 'var(--text-dim)',
                    fontSize: 10,
                    cursor: 'pointer',
                  }}
                >
                  Cancel
                </button>
                <button
                  onClick={handleCreate}
                  disabled={saving}
                  style={{
                    padding: '4px 12px',
                    background: 'rgba(255,215,0,0.15)',
                    border: '1px solid rgba(255,215,0,0.3)',
                    borderRadius: 4,
                    color: '#ffd700',
                    fontSize: 10,
                    fontWeight: 600,
                    cursor: saving ? 'wait' : 'pointer',
                  }}
                >
                  {saving ? 'Saving...' : 'Save'}
                </button>
              </div>
            </div>
          )}

          {/* Annotation list */}
          {annotations.length === 0 ? (
            <div style={{ padding: 12, textAlign: 'center', fontSize: 10, color: 'var(--text-dim)' }}>
              No annotations yet. Hover over the sky and mark interesting regions.
            </div>
          ) : (
            <div>
              <div style={{ fontSize: 9, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 6 }}>
                Saved ({annotations.length})
              </div>
              {annotations.map(ann => {
                const typeInfo = getTypeInfo(ann.type);
                return (
                  <div
                    key={ann.id}
                    style={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: 6,
                      padding: '6px 8px',
                      marginBottom: 3,
                      borderRadius: 5,
                      background: 'rgba(255,255,255,0.02)',
                      border: '1px solid transparent',
                      transition: 'all 0.15s',
                    }}
                  >
                    <span style={{ fontSize: 11, flexShrink: 0, marginTop: 1 }}>{typeInfo.icon}</span>
                    <button
                      onClick={() => onGotoAnnotation(ann.ra_center, ann.dec_center)}
                      style={{
                        flex: 1,
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        textAlign: 'left',
                        padding: 0,
                        minWidth: 0,
                      }}
                    >
                      <div style={{ fontSize: 10, fontWeight: 500, color: typeInfo.color }}>
                        {typeInfo.label}
                      </div>
                      {ann.notes && (
                        <div style={{
                          fontSize: 9,
                          color: 'var(--text-secondary)',
                          marginTop: 1,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}>
                          {ann.notes}
                        </div>
                      )}
                      <div className="mono" style={{ fontSize: 8, color: 'var(--text-dim)', marginTop: 1 }}>
                        {ann.ra_center.toFixed(2)}° {ann.dec_center >= 0 ? '+' : ''}{ann.dec_center.toFixed(2)}°
                      </div>
                    </button>
                    <button
                      onClick={() => onDeleteAnnotation(ann.id)}
                      style={{
                        background: 'none',
                        border: 'none',
                        color: 'var(--text-dim)',
                        cursor: 'pointer',
                        fontSize: 11,
                        padding: '0 2px',
                        opacity: 0.5,
                        flexShrink: 0,
                      }}
                      title="Delete annotation"
                    >
                      ×
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}
    </div>
  );
}
