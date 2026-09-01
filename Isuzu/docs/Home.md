# Isuzu vault

このフォルダは **Obsidian** で開くためのノート群です。数式はすべて `$`（インライン）と `$$`（ディスプレイ）です。GitHub 上では `$` が生に見えることがあります。Obsidian で `docs/` を vault として開いてください。

## 要件との対応

1. 数式は全部 Obsidian 向け Markdown（この vault とリポジトリ直下の `README.md`）。
2. 各アルゴリズムに文献を付け、オリジナルから進んだ／補った箇所は [[deviations]] に全部書いた。
3. ベータ・ベルヌーイ過程とノンパラメトリックベイズは [[npbayes]] と `src/npbayes/`。
4. 速度と精度の実測は [[benchmarks]]。

## 目次

- [[citations]] — 文献一覧（著者・年・誌名）
- [[deviations]] — オリジナル通りでない部分の台帳
- [[benchmarks]] — 典型問題の速度・精度
- [[rng]] — ChaCha8 / 分布サンプラー
- [[linalg-optimize]] — faer 分解と Nelder–Mead
- [[simulation]] — Itô スキームと駆動ノイズ
- [[models]] — 拡散・ジャンプ・Lévy・CARMA / COGARCH
- [[inference]] — QMLE / LSE / adaBayes / 変化点
- [[point-processes]] — ポアソン / Hawkes / CARMA-Hawkes / ACD
- [[filters]] — カルマン族と粒子フィルタ
- [[finance]] — Shreve 価格・ツリー・PDE・LSM・金利
- [[energy]] — Schwartz–Smith / Lucia–Schwartz / リアルオプション
- [[hft-control-malliavin]] — 高頻度・制御・Malliavin
- [[npbayes]] — DP / PY / Beta–Bernoulli / IBP / HDP

クレートの入口は [`README.md`](../README.md)（同じ数式規約）。
