# Soccer Backtest V12 Hardened Production

V9の検証済み時系列バックテストエンジンを土台に、V12のwalk-forward meta layerを追加したproduction版です。

## ユーザー操作
GitHubの **Actions → Soccer Backtest V12 - Hardened Production → Run workflow** を実行するだけです。通常はpush/scheduleでも自動実行されます。

## V12構成
- Football-Data.co.uk: FT結果 / 日程 / closing odds / 基本match stats
- Understat: xG / xGA
- SofaScore: 過去選手rating / player stats / 実績MOM
- 自前計算: Elo / form / home-away / rest / H2H / season transition
- ML: Logistic Regression / ExtraTrees / RandomForest / HistGradientBoosting
- Score: Dynamic Poisson + Dixon-Coles
- V9 base blend: 過去OOS実績ベースのML / market / Poisson blend
- V12 adaptive meta blend: Market / ML / Poissonを直近過去windowのLogLossで動的加重
- V12 meta-selection: V9 baseとadaptive blendを過去windowのLogLossだけで選択
- V12 temperature calibration: 現在の試合より前の履歴だけでtemperatureを選択
- chronological walk-forward validation + recency weighting

## リーク防止
- 当該試合の結果、stats、xG、ratingは予測後にstateへ反映
- 当該試合のlineupをMOM候補選定に使用しない
- closing oddsは試合前情報としてのみ使用
- V12 meta layerは現在行より前の予測・結果だけを使用
- ランダムsplitは禁止
- チーム名はcanonical keyで正規化し、危険な部分一致を禁止
- 欠損値を都合よく実績値で補完しない

## 運用ハードニング
- preflight → smoke → integrity/leakage audit → resumable state → backtest → V12 meta layer → output validation → artifact/repository persistence
- 30分枠を前提にcheckpoint/artifactを保存し、次回実行へ自動resume
- V12 artifactを優先し、初回は既存V9 artifactへfallback
- 完走時のみV9 engine completion markerとV12結果の両方を確認してschedule停止
- 結果CSVとcheckpointをmainへ永続化し、Actions artifactにも90日保存

## 出力
- `backtest_results_v12.csv`: V12最終確率・予測・confidence・meta mode・temperature
- `overall_summary_v12.csv`: Accuracy / LogLoss / Brier / MeanConfidence / AdaptiveShare
- V9の詳細なleague/season/score/MOM/model/data-coverage/feature-importance結果も保持

## 評価
Accuracyだけでなく、LogLoss / Brier / calibration / league-season別精度 / score Top-1/Top-3 / データ取得率を確認します。

「最強」「100%」は実戦100%的中を意味しません。V12は、利用可能な過去情報だけを使い、OOS性能が確認可能な範囲でモデル・校正・blendを強化する設計です。未検証の性能を捏造せず、バックテスト結果で採用可否を判断します。

## 注意
途中checkpointの指標は完成版の性能値と混同しません。データ欠損や外部API失敗があった場合は推測で埋めず、coverageとして記録します。
