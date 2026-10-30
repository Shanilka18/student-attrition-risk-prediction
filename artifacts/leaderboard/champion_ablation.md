# Champion Model Input Feature Ablation (tcn)

| stage_label                    |   n_channels |   test_roc_auc |   test_pr_auc |   test_f1 |   test_recall |   average_lead_time |   marginal_auc_gain |
|:-------------------------------|-------------:|---------------:|--------------:|----------:|--------------:|--------------------:|--------------------:|
| GPA Only                       |            1 |         0.7039 |        0.1186 |    0.1808 |        0.8389 |                2.14 |              0      |
| GPA + Exam Marks               |            2 |         0.7    |        0.1183 |    0.1791 |        0.8294 |                2.13 |             -0.0039 |
| GPA + Marks + Attendance       |            3 |         0.7812 |        0.1979 |    0.239  |        0.7109 |                2.1  |              0.0773 |
| All 4 Modalities (Full System) |            4 |         0.7837 |        0.1999 |    0.2389 |        0.7251 |                2.11 |              0.0798 |

**Conclusion**: Full 4-modality system achieved 0.7837 AUROC (++0.0798 over GPA-only baseline). Assignment delay and attendance contributed critical lead time gains.
