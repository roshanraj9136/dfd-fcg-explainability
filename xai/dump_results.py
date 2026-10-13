import json
R = json.load(open('xai/out/report.json'))
D = json.load(open('xai/out/derived.json'))
RS = json.load(open('xai/out/robustness_summary.json'))
r, f = R['videos']['000'], R['videos']['000_003']

print('--- HEAD PROBABILITIES ---')
for k in ['spatial_mean', 'temporal_mean', 'fusion_mean', 'final_mean',
          'final_std', 'final_min', 'final_max', 'final_median']:
    print(f'{k:16s} REAL {r["prob"][k]:.4f}   FAKE {f["prob"][k]:.4f}')

print('\n--- LOGIT SPACE (d = logit FAKE - logit REAL) ---')
for k in ['d_spatial_mean', 'd_temporal_mean', 'd_fusion_mean',
          'fusion_spatial_term_mean', 'fusion_temporal_term_mean',
          'pct_spatial_mean', 'pct_temporal_mean']:
    print(f'{k:28s} REAL {r["branch_logit"][k]:+.4f}   FAKE {f["branch_logit"][k]:+.4f}')

print('\n--- SHAPLEY ---')
for k in ['phi_spatial_mean', 'phi_temporal_mean', 'pct_spatial_mean',
          'pct_spatial_std', 'pct_temporal_mean', 'pct_temporal_std',
          'clips_spatial_dominant', 'clips_temporal_dominant']:
    print(f'{k:26s} REAL {r["branch_shapley"][k]}   FAKE {f["branch_shapley"][k]}')

print('\n--- FACIAL COMPONENTS ---')
for p in ['eyes', 'skin', 'nose', 'lips']:
    print(f'{p:6s} REAL {r["spatial_parts"]["contrib_mean"][p]:+.4f} '
          f'({r["spatial_parts"]["share_pct"][p]:5.2f}%)   '
          f'FAKE {f["spatial_parts"]["contrib_mean"][p]:+.4f} '
          f'({f["spatial_parts"]["share_pct"][p]:5.2f}%)   '
          f'delta {D["delta_parts"][p]:+.4f}')

print('\n--- LAYERS (top-5 by |contribution|) ---')
print('REAL spatial', r['layers']['spatial_top'], ' temporal', r['layers']['temporal_top'])
print('FAKE spatial', f['layers']['spatial_top'], ' temporal', f['layers']['temporal_top'])

print('\n--- ZONES (area-normalised |contribution| per patch) ---')
for z in ['eyes', 'brows', 'nose', 'mouth', 'jaw/contour', 'cheek/forehead', 'background']:
    print(f'{z:16s} REAL {D["zone_density"]["000"][z]:.5f}   '
          f'FAKE {D["zone_density"]["000_003"][z]:.5f}   '
          f'delta {D["zone_density_delta"][z]:+.5f}   '
          f'patches {D["zone_patch_counts"]["000"][z]:3d}')

print('\n--- COUNTERFACTUAL ---')
for v, nm in [('000', 'REAL'), ('000_003', 'FAKE')]:
    c = D['counterfactual'][v]
    print(f'{nm}: S-only {c["spatial_only_mean"]:.4f} ({c["spatial_only_verdict"]}) '
          f'T-only {c["temporal_only_mean"]:.4f} ({c["temporal_only_verdict"]}) '
          f'full {c["full_mean"]:.4f} ({c["full_verdict"]}) '
          f'agree {c["clip_branch_agreement_pct"]:.1f}%')

print('\n--- ROBUSTNESS ---')
print(f'conditions {RS["n_conditions"]}  FP {RS["false_positives"]}  FN {RS["false_negatives"]}')
print(f'{"cond":<9}{"param":>7}{"REALp":>8}{"verdict":>9}{"FAKEp":>8}{"verdict":>9}{"S%real":>8}{"S%fake":>8}')
for x in RS['rows']:
    print(f'{x["type"]+" L"+str(x["level"]):<9}{str(x["param"]):>7}{x["real_p"]:>8.3f}'
          f'{("FAKE" if x["real_p"]>=.5 else "REAL"):>9}{x["fake_p"]:>8.3f}'
          f'{("FAKE" if x["fake_p"]>=.5 else "REAL"):>9}'
          f'{x["real_pct_s"]:>8.1f}{x["fake_pct_s"]:>8.1f}')
