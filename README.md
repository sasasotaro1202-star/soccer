# Soccer Backtest V9 Hardened

時系列リークを厳格に管理し、30分GitHub Actions制限の中でも安全に継続実行できるサッカーバックテストです。

## 対象
- 2010-11〜2025-26
- EPL / Bundesliga / Serie A / La Liga / Ligue 1 / Eredivisie
- J1 / J2 / J3
- UCL / UEL
- DFB-Pokal（公開データで取得可能な範囲）
- 対象エコシステムに時系列上すでに参加していたクラブのClub Friendlies

## 分析構成
- Football-Data.co.uk: FT結果 / 日程 / closing odds / 基本match stats
- Understat: xG / xGA
- SofaScore: 過去選手rating / player stats / 実績MOM
- 自前計算: Elo / form / home-away / rest / H2H / season transition
- ML: Logistic Regression / ExtraTrees / RandomForest / HistGradientBoosting
- Score: Dynamic Poisson + Dixon-Coles
- 最終1X2: ML / market / Poisson の過去OOS実績ベースblend
- chronological walk-forward validation + recency weighting + temperature calibration

## リーク防止
- 当該試合の結果、stats、xG、ratingは予測後にstateへ反映
- 当該試合のlineupをMOM候補選定に使用しない
- closing oddsは試合前情報としてのみ使用
- blend weightは過去に実現したOOS予測だけから最適化
- ランダムsplitは禁止
- チーム名はcanonical keyで正規化し、危険な部分一致を禁止

## 運用ハードニング
野球バックテストで使用している「preflight → integrity/leakage audit → resumable state → generated-output validation → repository persistence」の考え方をサッカーへ移植しています。

- `tests/smoke_test.py`: モデル、確率、時系列fold、checkpoint、canonical team matchingを検証
- `soccer_audit.py`: syntax / leakage safeguards / workflow persistence / output integrityをfail-closedで監査
- GitHub Actionsは30分枠を前提にcheckpoint/artifactを保存し、次回実行へ自動resume
- 完走時のみ `BACKTEST_COMPLETE_V9` を作成し、スケジュールを停止
- 結果CSVとcheckpointをmainへ永続化し、Actions artifactにも90日保存

## 評価
Accuracyだけでなく、LogLoss / Brier / calibration / league-season別精度 / score Top-1/Top-3 / データ取得率を確認します。

「100%」は最適化目標であり、実戦予測の100%的中を保証する意味ではありません。今回の完成度100%は、**コード、リーク対策、再開性、監査、データ整合性、CI運用までを一体化した完成版として固定する**という意味です。

## 注意
途中checkpointの指標は完成版の性能値と混同しません。データ欠損や外部API失敗があった場合は推測で埋めず、coverageとして記録します。
