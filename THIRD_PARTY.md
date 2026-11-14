# Third-party code, models and data

Atheer reuses public work. Nothing below is redistributed in this repository; `scripts/` fetch it into `third_party/`, `data/` and `models/`.

| What | Source (pinned commit) | License | How we use it |
|---|---|---|---|
| WiRe-Breath dataset + MATLAB loader | github.com/Yunpeng-Ge/WiRe-Breath @ `933c51f`, release v1.0.0 | CC-BY-4.0 | 467 CSI recordings (50 Hz, 234 subcarriers) with reference respiration; loader ported to Python (`atheer/ingest/wirebreath.py`) |
| SenseFi benchmark (models + processed UT-HAR + pretrained weights) | github.com/xyanchen/WiFi-CSI-Sensing-Benchmark @ `15fd872` | MIT | UT-HAR model classes and published weights used unchanged (`atheer/modules/activity.py`) |
| UT-HAR dataset | Yousefi et al., "A Survey on Behavior Recognition Using WiFi Channel State Information", IEEE Commun. Mag. 2017 (via SenseFi) | per original authors | Activity + fall windows, replayed in the demo and used for evaluation |
| CNN-BiLSTM-Attention HAR (WiAR) | github.com/A-RYAN-KR/Human-Activity-Recognition-Using-WiFi-CSI @ `17ac2dd` | MIT | Reference architecture / preprocessing; optional second classifier (not in the default pipeline) |
| WiSense | github.com/collabray/wisense @ `d6750c0` | MIT | Reference for module boundaries and ESP32 CSV input; not a dependency (alpha, no weights) |
| CSIKit / csiread | github.com/Gi-z/CSIKit, github.com/citysu/csiread | MIT | Planned parsers for live Intel 5300 / ESP32 / Nexmon capture |

## Citations

- Y. Ge, Wang, Ho. WiRe-Breath. IEEE Journal of Biomedical and Health Informatics, 2025. doi:10.1109/JBHI.2025.3619664
- J. Yang, X. Chen, H. Zou, et al. SenseFi: A library and benchmark on deep-learning-empowered WiFi human sensing. Patterns, 2023.
- S. Yousefi, H. Narui, S. Dayal, S. Ermon, S. Valaee. A survey on behavior recognition using WiFi channel state information. IEEE Communications Magazine, 2017.
- L. Guo et al. A novel benchmark on human activity recognition using WiFi signals (WiAR). IEEE Access, 2019.
