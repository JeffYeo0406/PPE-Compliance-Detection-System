import cv2

from inference import CLASS_COLORS_RGB, CLASS_NAMES, CRITICAL_CLASSES, NEUTRAL_CLASSES, WARNING_CLASSES


def draw_detections(image_rgb, detections):
    annotated = image_rgb.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color = CLASS_COLORS_RGB.get(det['class_name'], (160, 160, 160))
        label = f"{det['class_name']} {det['confidence']:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 3, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def compute_stats(detections):
    counts = {name: 0 for name in CLASS_NAMES}
    for det in detections:
        class_name = det.get('class_name')
        if class_name in counts:
            counts[class_name] += 1

    person_present = counts['Person'] > 0
    emergency = counts['Fall-Detected'] > 0

    # Direct violation detection via NO-* classes
    critical_violations_found = counts['NO-Hardhat'] + counts['NO-Safety Vest']
    critical_present = counts['Hardhat'] + counts['Safety Vest']

    # Also flag as missing if person present but no compliant PPE detected
    critical_ok = 2  # Hardhat + Safety Vest
    critical_missing = []
    if person_present:
        if counts['Hardhat'] == 0 and counts['NO-Hardhat'] == 0:
            critical_missing.append('Hardhat')
        if counts['Safety Vest'] == 0 and counts['NO-Safety Vest'] == 0:
            critical_missing.append('Safety Vest')
    critical_ok -= len(critical_missing)

    warning_missing = []
    if person_present:
        warning_missing = [name for name in sorted(WARNING_CLASSES) if counts[name] == 0]

    warning_present = [name for name in sorted(WARNING_CLASSES) if counts[name] > 0]
    has_violation = critical_violations_found > 0 or len(critical_missing) > 0
    compliance_rate = 100.0 if not person_present else round((critical_ok / 2) * 100, 1)

    return {
        'counts': counts,
        'person_present': person_present,
        'emergency': emergency,
        'critical_missing': critical_missing,
        'critical_violations_found': critical_violations_found,
        'critical_present': critical_present,
        'warning_missing': warning_missing,
        'warning_present': warning_present,
        'critical_violations': critical_violations_found + len(critical_missing),
        'warning_alerts': len(warning_missing),
        'critical_ok': critical_ok,
        'critical_total': 2,
        'compliance_rate': compliance_rate,
        'has_violation': has_violation,
        'active_classes': sum(1 for count in counts.values() if count > 0),
    }


def _pill(text: str, style: str) -> str:
    styles = {
        'green': 'background:#1a3d2e;color:#5DCAA5;border:0.5px solid #1D9E75;',
        'red': 'background:#3a1a1a;color:#E24B4A;border:0.5px solid #E24B4A;',
        'blue': 'background:#1a2640;color:#7ab3e0;border:0.5px solid #185FA5;',
        'amber': 'background:#3a2b10;color:#BA7517;border:0.5px solid #BA7517;',
        'muted': 'background:#2a2a3a;color:#8888a0;border:0.5px solid #444460;',
    }
    css = styles.get(style, styles['muted'])
    return (
        f'<span style="font-size:11px;padding:2px 9px;border-radius:4px;'
        f'font-weight:500;{css}">{text}</span>'
    )


def render_header_html(stats=None, model_label: str = 'DEMO'):
    if stats is None or not stats['has_violation']:
        badge = (
            '<span style="display:flex;align-items:center;gap:6px;font-size:12px;'
            'background:#0e2a1e;color:#5DCAA5;padding:6px 14px;border-radius:20px;'
            'border:0.5px solid #1D9E75">● Ready</span>'
        )
    else:
        badge = (
            f'<span style="display:flex;align-items:center;gap:6px;font-size:12px;'
            f'background:#2d1212;color:#E24B4A;padding:6px 14px;border-radius:20px;'
            f'border:0.5px solid #E24B4A">● {stats["critical_violations"]} critical issue(s)</span>'
        )

    return f"""
<div style="display:flex;align-items:center;justify-content:space-between;
  padding:14px 20px;background:#16162a;border-radius:10px 10px 0 0;
  border-bottom:0.5px solid #2a2a40;margin-bottom:0">
  <div style="display:flex;align-items:center;gap:12px">
    <div style="width:32px;height:32px;border-radius:8px;background:#1e2d48;
      display:flex;align-items:center;justify-content:center;font-size:16px">🦺</div>
    <div>
      <div style="font-size:15px;font-weight:500;color:#e0e0f0">Real-time PPE detection</div>
      <div style="font-size:11px;color:#7070a0;margin-top:1px">
        Real-time safety monitoring · {model_label}</div>
    </div>
  </div>
  {badge}
</div>"""


def render_metrics_html(stats: dict, inference_ms: float) -> str:
    fps = round(1000 / inference_ms) if inference_ms > 0 else 0
    critical_str = f"{stats['critical_ok']} / {stats['critical_total']}"

    return f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:4px">
  <div style="background:#0e0e1c;border-radius:8px;padding:10px 13px">
    <div style="font-size:22px;font-weight:500;color:{'#E24B4A' if stats['critical_violations'] else '#5DCAA5'};line-height:1.1">{stats['critical_violations']}</div>
    <div style="font-size:11px;color:#7070a0;margin-top:4px">Critical issues</div>
  </div>
  <div style="background:#0e0e1c;border-radius:8px;padding:10px 13px">
    <div style="font-size:22px;font-weight:500;color:#e0e0f0;line-height:1.1">{inference_ms:.0f} ms</div>
    <div style="font-size:11px;color:#7070a0;margin-top:4px">Inference · {fps} FPS</div>
  </div>
  <div style="background:#0e0e1c;border-radius:8px;padding:10px 13px">
    <div style="font-size:22px;font-weight:500;color:#BA7517;line-height:1.1">{stats['warning_alerts']}</div>
    <div style="font-size:11px;color:#7070a0;margin-top:4px">Warnings</div>
  </div>
  <div style="background:#0e0e1c;border-radius:8px;padding:10px 13px">
    <div style="font-size:22px;font-weight:500;color:#5DCAA5;line-height:1.1">{critical_str}</div>
    <div style="font-size:11px;color:#7070a0;margin-top:4px">Critical compliance · {stats['compliance_rate']}%</div>
  </div>
</div>"""


def _class_row(label: str, pill: str) -> str:
    return (
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'padding:7px 10px;border-radius:7px;background:#1e1e2e;margin-bottom:5px">'
        f'<span style="font-size:13px;color:#b0b0c8">{label}</span>{pill}</div>'
    )


def render_report_html(stats: dict) -> str:
    # Critical PPE rows: Hardhat + Safety Vest with direct NO-* detection
    hh = stats['counts']['Hardhat']
    no_hh = stats['counts']['NO-Hardhat']
    sv = stats['counts']['Safety Vest']
    no_sv = stats['counts']['NO-Safety Vest']

    def _critical_pill(name, compliant, violation):
        if violation > 0:
            return _pill(f'✗ VIOLATION ({violation})', 'red')
        if compliant > 0:
            return _pill(f'✓ {compliant} detected', 'green')
        return _pill('not detected', 'muted')

    critical_rows = [
        ('Hardhat',     _critical_pill('Hardhat', hh, no_hh)),
        ('Safety Vest', _critical_pill('Safety Vest', sv, no_sv)),
    ]

    warning_rows = [
        (name, _pill(f"{stats['counts'][name]} detected", 'amber') if stats['counts'][name] > 0 else _pill('not detected', 'muted'))
        for name in sorted(WARNING_CLASSES)
    ]

    neutral_rows = [
        (name, _pill(f"{stats['counts'][name]} detected", 'blue') if stats['counts'][name] > 0 else _pill('0', 'muted'))
        for name in sorted(NEUTRAL_CLASSES)
        if name != 'Person'
    ]

    # Emergency row
    if stats.get('emergency'):
        emergency_row = _class_row('⚠ Fall-Detected', _pill(f'{stats["counts"]["Fall-Detected"]} instances', 'red'))
    else:
        emergency_row = ''

    rows_html = emergency_row
    rows_html += ''.join(_class_row(label, pill) for label, pill in critical_rows)
    rows_html += ''.join(_class_row(label, pill) for label, pill in warning_rows)

    summary_bits = [
        f"<strong>Persons:</strong> {stats['counts']['Person']}",
        f"<strong>Active classes:</strong> {stats['active_classes']}",
    ]
    if stats['critical_violations_found']:
        summary_bits.append(f"<strong>Violations:</strong> NO-Hardhat={no_hh}, NO-Safety Vest={no_sv}")
    if stats['critical_missing']:
        summary_bits.append(f"<strong>Missing:</strong> {', '.join(stats['critical_missing'])}")

    if stats.get('emergency'):
        status_html = (
            '<div style="display:flex;align-items:center;gap:7px;font-size:12px;'
            'background:#3a1010;color:#FF4444;padding:8px 12px;border-radius:7px;'
            'margin-top:6px;border:0.5px solid #FF4444">'
            '🚨 EMERGENCY — Fall detected</div>'
        )
    elif stats['has_violation']:
        status_html = (
            '<div style="display:flex;align-items:center;gap:7px;font-size:12px;'
            'background:#2d1212;color:#E24B4A;padding:8px 12px;border-radius:7px;'
            'margin-top:6px;border:0.5px solid #E24B4A">'
            f'⚠ {stats["critical_violations"]} critical violation(s) — action required</div>'
        )
    elif stats['warning_alerts'] > 0:
        status_html = (
            '<div style="display:flex;align-items:center;gap:7px;font-size:12px;'
            'background:#2a2010;color:#BA7517;padding:8px 12px;border-radius:7px;'
            'margin-top:6px;border:0.5px solid #BA7517">'
            '! PPE warnings detected — review recommended</div>'
        )
    else:
        status_html = (
            '<div style="display:flex;align-items:center;gap:7px;font-size:12px;'
            'background:#0e2a1e;color:#5DCAA5;padding:8px 12px;border-radius:7px;'
            'margin-top:6px;border:0.5px solid #1D9E75">'
            '✓ All critical PPE compliant</div>'
        )

    neutral_html = ''.join(_class_row(label, pill) for label, pill in neutral_rows)

    return f"""
{rows_html}
<div style="margin-top:10px;text-align:center">
  <div style="font-size:32px;font-weight:500;color:#e0e0f0;line-height:1">{stats['compliance_rate']}%</div>
  <div style="font-size:12px;color:#7070a0;margin-top:4px">Critical compliance rate</div>
  <div style="height:6px;background:#2a2a3a;border-radius:3px;margin-top:10px;overflow:hidden">
    <div style="height:6px;background:{'#1D9E75' if not stats['has_violation'] else '#E24B4A'};border-radius:3px;width:{int(stats['compliance_rate'])}%;transition:width .5s ease"></div>
  </div>
</div>
<div style="margin-top:10px;color:#b0b0c8;font-size:12px;line-height:1.5">
  {'<br>'.join(summary_bits)}
</div>
{status_html}
<div style="margin-top:12px;color:#7070a0;font-size:11px">Context</div>
{neutral_html}
"""


def render_footer_html() -> str:
    return f"""
<div style="display:flex;align-items:center;gap:0;padding:10px 20px;
  background:#16162a;border-radius:0 0 10px 10px;
  border-top:0.5px solid #2a2a40;margin-top:0">
  <div style="text-align:center;padding-right:16px">
    <div style="font-size:13px;font-weight:500;color:#e0e0f0">14 classes</div>
    <div style="font-size:10px;color:#7070a0;margin-top:1px">Dataset</div>
  </div>
  <div style="width:0.5px;height:24px;background:#2a2a40;margin:0 16px"></div>
  <div style="text-align:center">
    <div style="font-size:13px;font-weight:500;color:#e0e0f0">{len(CRITICAL_CLASSES)} critical</div>
    <div style="font-size:10px;color:#7070a0;margin-top:1px">Compliance</div>
  </div>
  <div style="width:0.5px;height:24px;background:#2a2a40;margin:0 16px"></div>
  <div style="text-align:center">
    <div style="font-size:13px;font-weight:500;color:#e0e0f0">{len(WARNING_CLASSES)} warning</div>
    <div style="font-size:10px;color:#7070a0;margin-top:1px">Advisories</div>
  </div>
  <div style="width:0.5px;height:24px;background:#2a2a40;margin:0 16px"></div>
  <div style="text-align:center">
    <div style="font-size:13px;font-weight:500;color:#e0e0f0">{len(NEUTRAL_CLASSES)} neutral</div>
    <div style="font-size:10px;color:#7070a0;margin-top:1px">Context</div>
  </div>
  <div style="margin-left:auto;font-size:11px;color:#4a4a60">
    BERR 4743 · Computer Vision and Pattern Recognition
  </div>
</div>"""
