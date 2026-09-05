# Changelog

このファイルには利用者に影響する変更を記録します。

## Unreleased

### Added

- `tsutsumi` を追加し、行列・分解・最小二乗・特殊関数・Nelder–Mead・正規求積を
  一つの独立検証可能な共通核へ集約しました。旧 kanimiso の同名モジュールは再公開です。
- `number-ruler` を追加しました。LM / canonical GLM、ランダム切片 LMM / GLMM、
  区分線形 LAM / GAM、線形予測子の interventional SHAP を、結果固有の注釈付きで
  提供します。混合・加法モデルは Experimental。未実装の推論を便宜的な p 値で埋めません。
- `tsutsumi` の依存なし行列契約だけに依存する `oldwood` crate を追加し、weighted CART classifier / regressor、
  runtime criterion、`predict_proba`、`apply`、arena inspection、feature importance、
  forest 用の検証済み split-candidate 境界を実装しました。
- 通常 CART を `oldwood` と共有する `mayoi-no-mori` crate を追加しました。
  random forest、ExtraTrees、通常 GBDT、discrete SAMME、AdaBoost.R2、Isolation
  Forest、histogram/Newton boosting、ordered categorical statistics を提供します。
  ensemble は Experimental で、LightGBM / CatBoost 系は実装範囲と非互換点を
  README に明記します。
- ensemble の乱数に、Isuzu の `amatsuki::ChaCha8Rng` を core-only で共有しました。
- scikit-learn 1.7.2 由来の weighted CART golden を追加し、Gini、entropy、
  squared error の probe 出力を Rust から再生します。
- statsmodels 由来の LM / GLM / LMM と、独立適応積分による GLMM oracle、
  SHAP 全 coalition 列挙、加法性・群再ラベル・失敗境界のテストを追加しました。

### Changed

- v0.1 の数値パラメータ別に生成された公開型を削除し、v0.2 の実行時パラメータ API
  へ統合しました。削除名と移行先は
  [`docs/dropped_v0_1.md`](https://github.com/yaruodemanabu/kanimiso/blob/main/docs/dropped_v0_1.md)
  を参照してください。
- README を利用者向けの導入・対応範囲・品質契約を先に示す構成へ変更しました。
- `kanimiso::tree` と `kanimiso::histgb` は品質契約アダプターへ縮小し、CART と ensemble
  の split / arena / traversal 実装を `oldwood` / `mayoi-no-mori` に一本化しました。
  これは source-compatible な置換ではありません。`predict_proba_row` と Isolation
  Forest の直接 helper は `Session` を受け取り `Result<Qualified<_>>` を返すようになり、
  `IsolationForest::new` の引数と `RandomTreesEmbedding` は削除しました。
- 仕様適合を検証できなかった legacy SAMME.R の学習経路を削除し、選択時に明示的な
  失敗を返します。
- standalone crate の公開順序は root README の依存関係に従います。

### Fixed

- 正規分布 p 値の裾を直接計算し、`1 - CDF` による小確率の消失を防ぎました。
  `logsumexp` は最大値が無限でも NaN を隠しません。
- 共通 SVD は分解済みの特異値を使い、同じ行列を再分解しません。
- Histogram boosting は集約勾配・Hessian に基づく L1/L2 正則化 gain で split を選択し、
  bin の prefix/suffix で候補評価します。通常 CART とは異なる Newton 目的関数です。

- Student-t と F 分布の上側確率を補集合から直接計算し、極小 p 値が `1 - CDF` の
  桁落ちでゼロになる問題を修正しました。

### CI

- GitHub Actions を read-only 権限、immutable SHA、lockfile 固定、timeout 付きにし、
  line coverage と既存巨大ファイルの行数を減少専用 ratchet にしました。
