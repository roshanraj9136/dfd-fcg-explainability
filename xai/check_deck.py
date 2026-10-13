r"""Validate deck layout: off-slide shapes, overlaps, and text overflow estimates."""
from pptx import Presentation
from pptx.util import Emu, Inches

prs = Presentation("DFD_FCG_Explainability.pptx")
SW, SH = prs.slide_width, prs.slide_height
print(f"slide size: {SW/914400:.3f} x {SH/914400:.3f} in   slides: {len(prs.slides._sldIdLst)}")
problems = []

CH_W = {11: 0.0735, 11.5: 0.0768, 12: 0.080, 12.5: 0.0834, 13: 0.0868,
        13.5: 0.0901, 14.5: 0.0968, 16: 0.1068, 21: 0.140, 27: 0.180,
        32: 0.2136, 10.5: 0.070, 9.2: 0.0614, 9.8: 0.0654}

for si, slide in enumerate(prs.slides, 1):
    issues = []
    for sh in slide.shapes:
        if sh.left is None or sh.top is None:
            continue
        l, t = sh.left, sh.top
        w = sh.width or 0
        h = sh.height or 0
        # off-slide
        if l < -9144 or t < -9144 or l + w > SW + 9144 or t + h > SH + 9144:
            issues.append(f"OFF-SLIDE {sh.shape_type} name={sh.name!r} "
                          f"L={l/914400:.2f} T={t/914400:.2f} "
                          f"R={(l+w)/914400:.2f} B={(t+h)/914400:.2f}")
        # crude text overflow estimate for textboxes
        if sh.has_text_frame and sh.text_frame.text.strip():
            box_w_in = w / 914400
            total_lines = 0
            max_size = 0
            for p in sh.text_frame.paragraphs:
                txt = "".join(r.text for r in p.runs)
                if not txt:
                    total_lines += 1
                    continue
                size = max((r.font.size.pt for r in p.runs if r.font.size), default=12)
                max_size = max(max_size, size)
                cw = CH_W.get(size, size * 0.0067)
                cap = max(int(box_w_in / cw), 1)
                total_lines += max(1, -(-len(txt) // cap))
            est_h = total_lines * (max_size * 1.33 / 72) * 1.18
            if est_h > h / 914400 + 0.30:
                issues.append(f"TEXT OVERFLOW name={sh.name!r} est={est_h:.2f}in "
                              f"box={h/914400:.2f}in lines={total_lines} "
                              f"text={sh.text_frame.text[:45]!r}")
    # picture bounds
    for sh in slide.shapes:
        if sh.shape_type == 13:
            b = (sh.top + sh.height) / 914400
            if b > SH / 914400 + 0.02:
                issues.append(f"PICTURE past bottom: bottom={b:.2f}in")
    if issues:
        problems.append((si, issues))

if not problems:
    print("\nLAYOUT OK: no off-slide shapes, no estimated text overflow.")
else:
    for si, iss in problems:
        print(f"\n--- slide {si} ---")
        for i in iss:
            print("   ", i)
print(f"\nslides with issues: {len(problems)} / {len(prs.slides._sldIdLst)}")
