# Changelog

このファイルには利用者に影響する変更を記録します。

## Unreleased

### Changed

- v0.1 の数値パラメータ別に生成された公開型を削除し、v0.2 の実行時パラメータ API
  へ統合しました。削除名と移行先は
  [`docs/dropped_v0_1.md`](https://github.com/yaruodemanabu/kanimiso/blob/main/docs/dropped_v0_1.md)
  を参照してください。
- README を利用者向けの導入・対応範囲・品質契約を先に示す構成へ変更しました。

### Fixed

- Student-t と F 分布の上側確率を補集合から直接計算し、極小 p 値が `1 - CDF` の
  桁落ちでゼロになる問題を修正しました。

### CI

- GitHub Actions を read-only 権限、immutable SHA、lockfile 固定、timeout 付きにし、
  line coverage と既存巨大ファイルの行数を減少専用 ratchet にしました。
