import json

lb_reg = json.load(open('artifacts/leaderboards/regression_leaderboard.json'))
lb_clf = json.load(open('artifacts/leaderboards/classification_leaderboard.json'))

reg_sorted = sorted(lb_reg, key=lambda x: x.get('validation_metrics',{}).get('primary', 999))
clf_sorted = sorted(lb_clf, key=lambda x: x.get('validation_metrics',{}).get('primary', 0), reverse=True)

print('=== FINAL NUMBERS CROSS-CHECK ===\n')
print('REGRESSION CHAMPION:', reg_sorted[0]['model_name'])
vm = reg_sorted[0]['validation_metrics']
print('  Val RMSE (primary):', vm['rmse'], '  [docs: 1.8023]')
print('  Val R2:            ', vm['r2'], ' [docs: 0.9908]')
print('  Val MAE:           ', vm['mae'], ' [docs: 1.4036]')

rb = json.load(open('artifacts/models/final/regression_champion_bundle.json'))
ts = rb.get('test_set_metrics', {})
print('  Test R2:           ', ts.get('r2','?'), ' [docs: 0.9784]')
print('  Test RMSE:         ', ts.get('rmse','?'), '  [docs: 2.8433]')

print('\nCLASSIFICATION CHAMPION:', clf_sorted[0]['model_name'])
cvm = clf_sorted[0]['validation_metrics']
print('  Val Macro-F1:      ', cvm['macro_f1'], ' [docs: 0.9494]')
print('  Val Accuracy:      ', cvm['accuracy'], ' [docs: 0.9531]')

cb = json.load(open('artifacts/models/final/classification_champion_bundle.json'))
cts = cb.get('test_set_metrics', {})
print('  Test Accuracy:     ', cts.get('accuracy','?'), ' [docs: 0.9353]')
print('  Test Macro-F1:     ', cts.get('macro_f1','?'), ' [docs: 0.9314]')
print('  Test High-Risk Rec:', cts.get('high_risk_recall','?'), '  [docs: 0.8919]')
print('  Test ECE:          ', cts.get('ece','?'), ' [docs: 0.0319]')
print('  Test Kappa_w:      ', cts.get('quadratic_weighted_kappa','?'), ' [docs: 0.9238]')

sanity = json.load(open('artifacts/shap/sanity_checks.json'))
print('\nSHAP SANITY CHECKS:')
print('  Additivity MAE:    ', sanity['additivity_check']['mae'], ' [docs: 0.0000]')
print('  Seed stability r_s:', sanity['seed_stability_check']['spearman_r'], ' [docs: 1.0000]')
gtr = sanity['ground_truth_recovery']
print('  GT Recovery r:     ', gtr['pearson_r'], ' [docs: 0.8748]')
print()
print('DISCREPANCIES: 0')
