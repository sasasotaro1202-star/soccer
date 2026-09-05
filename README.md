# Soccer Backtest V9 FINAL

最終版の時系列リーク防止型サッカーバックテスト。

## 対象
- 2010-11〜2025-26
- EPL / Bundesliga / Serie A / La Liga / Ligue 1 / Eredivisie
- J1 / J2 / J3
- UCL / UEL
- DFB-Pokal（公開データで取得可能な範囲）
- 対象クラブのClub Friendlies（SofaScore取得可能な範囲）

## モデル
- Closing market probability
- Competition Elo + global cross-competition Elo
- Rolling / EMA team form
- xG / xGA
- Shots / SOT / corners
- Home/Away split
- Rest days
- H2H
- Player-history style features
- Dynamic Poisson + Dixon-Coles
- Logistic Regression / ExtraTrees / RandomForest / HistGradientBoosting
- Chronological walk-forward validation
- Recency weighting
- Temperature calibration
- ML / market / Poisson blend optimization
- MOMは試合前に蓄積された選手履歴だけを候補生成に使用

## V9で修正した点
- Logistic Pipelineの sample_weight を clf__sample_weight として正しく渡す
- OOS calibrationでモデルごとのvalidation targetを正しく対応させる
- H2Hの二重追加を修正
- 公式MOMが取得できない試合を「最高rating=MOM」として扱わない
- checkpointをgzip圧縮
- checkpoint内の学習履歴を最新MAX_TRAINに制限
- 学習済みmodel bundleをcheckpointに保存せず、Resume時に再構築してサイズを削減
- V8と別のcheckpoint / artifact / output名を使用
- Safe Stop時に BACKTEST FINISHED と誤表示しない
- 完走時だけ BACKTEST_COMPLETE_V9 を作成
- GitHub ActionsをNode 24系の現行majorへ更新
- preflight smoke testを追加

## 重要
「100%」は最適化目標であり、精度保証ではありません。
最終評価は全期間完走後のAccuracy / LogLoss / Brier / calibration / league-season別結果で行います。

## GitHub
.github/workflows/soccer-backtest.yml を使用してください。
古いV5/V7/V8のcheckpointはV9では使用しません。
