# Soccer Backtest V12 FINAL

時系列リークを厳格に管理した、30分GitHub Actions向けのサッカーバックテスト。

## 対象
- 2010-11〜2025-26
- EPL / Bundesliga / Serie A / La Liga / Ligue 1 / Eredivisie
- J1 / J2 / J3
- UCL / UEL
- DFB-Pokal（公開データで取得可能な範囲）
- 対象エコシステムに時系列上すでに参加していたクラブのClub Friendlies

## V12の主な改善
- resume時にgroup/cadence countersを再構築し、途中再開による検証スケジュールの変化を防止
- DFB-Pokalの対象判定をその試合日以前のBundesliga初出場に限定
- 過去の選手データに指数減衰を適用し、移籍・離脱後の古い選手状態が現チーム特徴を支配するのを防止
- MOM候補にも180日間のstaleness guardを適用し、過去所属選手の混入を抑制
- Logistic Pipelineのsample_weightを正しく適用
- chronological OOS validationをモデルごとに整合
- recency weighting + temperature calibration
- OOS実績だけでML/market/Poisson blendを最適化
- 未学習期間のfallback予測をblend最適化から除外
- H2H二重追加を排除
- 公式MOMが明示された試合だけMOM評価
- 試合前stateと試合後stateを完全分離
- クラブ名のcanonical keyを強化し、危険な部分一致を廃止
- DFB-Pokal / Friendliesの対象クラブ選定を試合日ベースにして将来参加情報の混入を防止
- checkpointをgzip化し、学習履歴をMAX_TRAIN以内に制限
- model bundleはcheckpointに保存せずresume時に再構築
- V12専用checkpoint / artifact / output
- SAFE STOPとCOMPLETEを明確に分離
- GitHub Actionsのデータcacheと状態artifactを分離

## 評価
Accuracyだけでなく、LogLoss / Brier / calibration / league-season別精度 / score Top-1/Top-3を確認します。

「100%」は最適化目標であり、精度保証ではありません。

## Fresh start
V12は過去のV9/V10/V11 checkpointを再利用せず、履歴を最初から時系列順に再計算します。
