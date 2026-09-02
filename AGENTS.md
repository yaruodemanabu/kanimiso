---
aliases: [kanimiso v0.2 refactor policy, kanimiso AGENTS.md, kanimiso 再構成方針]
tags: [kanimiso, wormhole, coronel, jelly_wave, signlred, ojizou_san, rust, refactoring, numerical_quality, agent_handoff]
created: 2026-09-01
related: ["[[kanimiso]]", "[[wormhole]]", "[[signlred]]", "[[ojizou_san]]", "[[isuzu_audit]]", "[[math_coloring_rules]]"]
---

# kanimiso v0.2 再構成方針（AGENTS.md）

> **配置**: リポジトリルートに `AGENTS.md` として置く。`CLAUDE.md` は `AGENTS.md` へのシンボリックリンク、Cursor は `.cursor/rules/agents.mdc` から `@AGENTS.md` を参照させる。
> **対象**: kanimiso で作業するコーディングエージェント（Cursor / Codex / Claude Code）と人間レビュアー。
> **性格**: コーディング規約集ではない。「何を作らないか」「何を壊してよいか」「完了を何で測るか」の決定書。機械検査できる規則は §7 の CI と付録 A の lint に落とす。

---

## 0. 現状診断（2026-09-01、`main` = `99c46d0`、PR #4 マージ後）

数値はすべてクローンして実測した。推測値はない。

| 指標 | 実測 | 備考 |
|---|---:|---|
| Rust 総行数（workspace） | 964,218 | PR #3 時点の概算 96.2 万から更に増加 |
| `kanimiso/src/hmm.rs` | 568,846 行 | `pub struct` 4,956、`impl` 12,387、`#[test]` **5** |
| `kanimiso/src/online.rs` | 197,219 行 | `pub struct` 1,314、`#[test]` **4** |
| `kanimiso/src/tslearn.rs` | 55,376 行 | `#[test]` 4 |
| `kanimiso/src/stats.rs` / `tsa.rs` / `coverage.rs` | 32,560 / 21,193 / 18,658 行 | |
| `kanimiso` 公開項目（`pub struct/enum/fn/trait/type/const/mod`） | 19,739 | `cargo public-api` 相当値はもっと多い |
| `#[test]` 総数 | 186 | 96 万行に対して |
| `coverage::inventory()` 登録数 | 1,871 | 検証状態の区分なし |
| CI | **なし**（`.github/` 不在） | status check ゼロでマージされている |
| `.cargo/config.toml` | `RUST_MIN_STACK = 32 MiB` | テストを通すためのグローバル回避策 |
| toolchain / MSRV | `rust-toolchain.toml` = 1.98.0、workspace `rust-version` = 1.85 | 1.85 ジョブで workspace 全体を継続検証する |

### 0.1 冗長性の実体（`hmm.rs`）

- 型名の数値を潰すと 4,956 → **1,476 族**。うち `{Unit,Beta,Kumaraswamy,Exponentiated,Discrete}×{Cosine,Tsp}×{N}` とその `Fitted*` が **3,460 型**（Cosine は N=3..173 の 171 通り、Tsp は 175 通り）。
- 冪ごとに `log_cosN` / `cosN_cdf` / `log_unit_cN` / `log_beta_cN` / `log_kuma_cN` / `log_exp_cN` / `log_disc_cN` が **各 171 本**。`cosN_cdf` は三角級数の手展開（`cos40_cdf` で 36 行、係数はリテラル）。
- 各型に `Default` / `new` / `fit` alias / `Fitted*` / `log_emit_seq` / `decode` / `Predict` / `FitUnsupervised` の同一ボイラープレートが反復。docstring は「Distinct from [`X`]」で差分だけ主張する。
- 数値リテラル: `1e-15` が 3,840 回、`1e-12` が 2,535 回、`.clamp(` 3,071 回、`.max(1e` 1,189 回。ポリシー化されていない。

### 0.2 冗長性の実体（`online.rs`）

- `WindowLag1..WindowLag418`（418 型）: k 次後退差分。k≤52 は二項係数をリテラル展開（最大 `C(52,26) ≈ 5e14`）、k≥53 は差分を k 回反復。**k 次差分の丸め誤差増幅はおよそ 2^k·ε**（k=20 で 1e-10、k=52 で既に O(1)）。リテラル展開された上位型の時点で結果は数値的に無意味、418 次差分は統計的にも意味を持たない。
- `LogMinkowski1..356Anomaly`（357 型）: `p` を型に焼き込んだ k-NN 異常度。`wormhole::Metric::Minkowski(f64)` が既に同じ概念を実行時パラメータで持っている。

### 0.3 致命的な数値バグ（P0、`special.rs`）

`betainc_reg(a, b, x)` の補集合分枝（`x ≥ (a+1)/(a+b+2)`）が **ベータ関数 B(a,b) で 2 回割っている**。scipy との差分テスト（`special_oracle_check.py` 同梱、Rust を Python に逐語移植）:

| 呼び出し | kanimiso | scipy 1.17.1 |
|---|---:|---:|
| `betainc_reg(3, 3, 0.5)` | **−14.0** | 0.5 |
| `student_t_pvalue(1.0, df=30)` | **0.000000** | 0.325309 |
| `student_t_pvalue(1.5, df=10)` | **0.000000** | 0.164507 |
| `f_pvalue(2.0, 3, 20)` | **5.418** | 0.146439 |

影響: `student_t_cdf` / `f_cdf` → `linear_model.rs`（6 箇所）、`stats.rs`（32 箇所）、`iv.rs`（10 箇所）、`robust.rs`、`tsa.rs`、`ensemble.rs`、`feature.rs`。README 冒頭の `fitted.value.p_values` は **有意でない係数を p≈0 と報告**し、F 検定は p>1 を返す。`Qualified<T>` は「p 値が 1 を超えた」ことを検出していない。186 本のテストに scipy オラクルが 1 本も無いことの直接の結果。

**結論**: 問題は「機能不足」でも「行数」でもなく、**検証可能性の欠如**。修正方針はここから決まる。

---

## 1. 採用決定（D1–D11、adopted default）

| # | 決定 | 補足 |
|---|---|---|
| D1 | `main` を `generated-v0.1-archive` でタグ付けし凍結。`refactor/v0.2` で小さい核から再構成する。検証計画のない新規アルゴリズム追加は alpha.1 まで停止 | 採用済みの基盤（単一の最適化核など）はオラクル・分岐テストを先に用意する |
| D2 | **v0.1 との数値一致は受入条件ではない**。受入条件はオラクル（scipy / statsmodels / hmmlearn / river / 閉形式 / brute-force）との一致 | v0.1 は §0.3 のとおり誤りを含む。旧値へのパリティは誤りを固定する |
| D3 | 値で変わるものは**実行時パラメータ**。型・関数・モジュール名に数値パラメータを埋め込まない。マクロ・コード生成による行数削減は解決と認めない | 単相化・公開 API・ドキュメント増殖は残るため |
| D4 | 外部の科学技術計算クレートは **`ndarray` と `faer` のみ**。wormhole / coronel / jelly-wave / argmin 等は参照実装・オラクルには使えるが依存へ追加しない | Pure Rust と依存面積の固定を CI で検査する |
| D5 | `ndarray` は N 次元配列・軸・view、`faer` は行列所有・分解・固有値・線形方程式・最小二乗を担当する。同じ線形代数を両方で実装しない | `ndarray-linalg` / `ndarray-stats` / native BLAS は不採用 |
| D6 | `ndarray = 0.17.2` と `faer = 0.24.4` を全 workspace で単一版に固定する | CI で lockfile、feature、`cargo tree -i` を検査 |
| D7 | workspace MSRV は `Cargo.toml` の 1.85。CI に 1.85 ジョブを置き、上げる場合は全 workspace を同時に更新する | 依存の公称 MSRV ではなく workspace 全体の実ビルドで保証 |
| D8 | 数値しきい値は `signlred::Policy` に集約（`log_prob_floor`、`max_difference_order`、`underflow_guard` 等を追加）。floor / clamp / jitter / ridge は `NumericalCompromise` 記録必須。密度を計算してから `ln` を取らない | `Policy` は既に条件数・残差・VIF 等を持つ。別構造体を作らない |
| D9 | HMM は `Emission` trait を持つ**単一の** `HiddenMarkovModel<E>`。forward–backward / Viterbi / Baum–Welch は各 1 実装。分布族は 1 ファイル 1 族 | §5 |
| D10 | `coverage::inventory()` に `status: verified / experimental / generated / stub` を追加し、README の主張を `verified` の集合に連動させる | 「六つの Python ライブラリ相当」表記は alpha.1 まで撤回 |
| D11 | 最適化は実行時係数を持つ単一の Pure Rust 実装に集約する。Nelder–Mead は argmin の分岐を参照し、決定的 trace・性質・既知解で検証するが argmin には依存しない | 外側/内側収縮、shrink、非有限目的値、停止理由を個別に検査 |

---

## 2. 冗長性禁止規則（R1–R10）

エージェントがコードを**書く前**と**書いた後**に照合する規則。各規則に検出方法を付ける。付録 A の `scripts/lint_redundancy.sh` が CI で機械検査する部分は「lint」、人間レビューは「review」。

| # | 規則 | なぜ | 検出 |
|---|---|---|---|
| R1 | 公開識別子（`struct` / `enum` / `trait` / `fn` / `type` / `mod`）に**パラメータ値としての数値**を含めない。`WindowLag7` → `WindowLag { order }`、`cos40_cdf` → `cos_power_cdf(n)`、`Garch11` → `Garch { p, q }`。固有名詞の数値（`Exp3`, `Ucb1`, `Catch22`, `X13`, `HiveCoteV2`, `Pooled2Sls`, `Chi2`, `F1`, `R2`）は許可リストで例外 | 値の数だけ型が増え、API・単相化・docs・テストが線形に膨らむ。現在 5,237 件 | lint（ratchet、付録 A。許可リスト `scripts/lint_allowlist.txt`） |
| R2 | 1 概念 1 実装。同じ関数体で定数だけ違うもの、同じフィールド集合の構造体が 3 回以上現れるものは禁止 | コピペ変種はバグも複製する（§0.3 の修正が 171 箇所に散る） | review + `cargo clippy` の `similar_names` ではなく、PR 説明で「同型の既存実装は無いか」を明記させる |
| R3 | マクロ・`build.rs`・外部スクリプトによる型生成で R1/R2 を回避しない。ジェネリクスは**データ型**（`Emission`、観測型）に対して使い、**パラメータ値**に対して使わない（`const N: usize` も禁止） | 公開面積・単相化・rustdoc は生成後の姿で膨らむ | review（`macro_rules!` で `pub struct` を生成する PR は差し戻し） |
| R4 | 公開項目予算。`kanimiso` の `pub` 項目は alpha.1 で **1,000 以下**。各 PR は予算値（付録 A の `MAX_R4_PUB_ITEMS`）を下げることはできるが上げることはできない | 19,739 項目は誰も監査できない | lint（ratchet） |
| R5 | ファイル予算 **3,000 行**。超える場合は責務で分割（`hmm/forward_backward.rs` 等）。「後で分ける」は不可 | 56 万行のファイルはレビュー不能、IDE も止まる。現在 13 ファイルが超過 | lint（ratchet） |
| R6 | 数値核は workspace 内で 1 実装だけにする。距離 / pairwise / Gram / DTW / OT / 最適化をモデルごとに再実装しない。特殊関数は `special.rs` 一箇所、`logsumexp` は一箇所 | 同じ計算が 2 箇所にあると片方だけ直る | review + 共通モジュールへの呼び出しを grep で確認 |
| R7 | 数値リテラル（`1e-15` 等）を関数本体に書かない。`Policy` のフィールドを参照する。`x.max(1e-15).ln()` 型の**密度 floor は禁止**、対数領域で計算し support 外は `-∞`。`NaN` を 0 や 1 に黙って置換しない | 裾の対数尤度を定数に潰すと EM とモデル選択が歪む。現在 1,857 件 | lint（ratchet、`\.max\(1e-…\)\.ln\(\)` / `\.clamp\(0\.0, 1\.0 - 1e` を検出）+ review |
| R8 | テストは「性質」と「オラクル」で書く。生成型ごとの smoke test は書かない。数値カーネルには **オラクル一致テストとプロパティテストの両方**が必須 | 186 本で 96 万行が通るのは、テストがカーネルを見ていないから | review（PR テンプレートに「オラクル: 何と、どの許容差で」欄） |
| R9 | 許容差は**実測 × 3〜4 の余裕**で決め、実測値をテストのコメントに残す。恣意的な `1e-6` を書かない | 閾値の由来が追えないテストは回帰を検出しない | review |
| R10 | docstring の「Distinct from [`X`]」パターンを禁止。差分ではなく**パラメータの意味**を書く | 変種増殖のマーカー。現在 3,435 件 | lint（ratchet） |

### 2.1 書く前の 5 問（エージェント自問）

1. これはパラメータで表現できないか？（できるなら型を増やさない）
2. workspace の共通数値核 / `special.rs` に既にないか？参照実装と分岐が一致するか？
3. `pub` を増やす必要が本当にあるか？予算は？
4. 書こうとしている数値定数は `Policy` にあるか？無ければ追加して参照する
5. この計算のオラクルは何か？どの許容差で、なぜその値か？

`Foo2` / `FooV2` / `FooAlt` を書きたくなった時点で立ち止まる。

---

## 3. 科学技術計算クレートの境界

### 3.1 固定する直接依存

```toml
# workspace Cargo.toml
[workspace.dependencies]
ndarray = { version = "=0.17.2", default-features = false, features = ["std"] }
faer = { version = "=0.24.4", default-features = false, features = ["std", "linalg", "rayon"] }
```

- 外部の科学技術計算クレートはこの 2 つだけ。`ndarray-linalg`、`ndarray-stats`、`argmin`、wormhole ファミリーは直接・推移の設計依存へ追加しない。
- `ndarray` の `blas` feature、および OpenBLAS / LAPACK / MKL / Accelerate / Netlib の source/sys crate を lockfile に入れない。
- crates.io の版は完全固定する。git / branch / path patch による外部依存の差し替えは禁止する。
- `scripts/lint_dependencies.py` が manifest、lockfile、feature、native backend 不在、first-party `unsafe` 不在を検査する。

### 3.2 責務分担

| 領域 | 所有者 |
|---|---|
| N 次元配列、軸操作、slice、view、テンソル状データ | `ndarray` |
| 2 次元行列の所有、分解、固有値、線形方程式、最小二乗 | `faer` |
| 統計モデル、オンライン更新、状態空間、確率過程 | kanimiso の検証済み共通核 |
| 数値品質の判定・停止条件 | `signlred::Policy` / `Qualified<T>` |
| 反復・妥協・収束の記録 | `ojizou_san::Session` |

- `Matrix` は faer-backed のままにする。ndarray との変換が必要なら、コピーであることを名前に含めた 1 箇所の adapter に閉じる。内側ループで相互変換しない。
- ndarray 上に第二の線形代数層を作らない。分解や solver を ndarray 用に再実装せず faer へ明示的に渡す。
- 同じ距離、kernel、DP、最適化、特殊関数をモデルごとに複製しない。上位モデルは共通数値核を呼ぶ。

### 3.3 外部実装の参照方法

scipy / statsmodels / hmmlearn / river / POT / argmin / wormhole ファミリーは、依存ではなく仕様確認とオラクル生成に利用できる。実装時は次を守る。

1. 分岐条件、停止条件、失敗条件を先に表へ起こす。
2. Pure Rust の共通核を 1 実装だけ書く。
3. 決定的 JSON または手計算可能な小問題で trace / 結果を比較する。
4. 外部実装との差を意図的に変える場合は理由と回帰テストを残す。

Nelder–Mead では特に、反射・拡大・外側収縮・内側収縮・shrink を別々に検証する。`argmin` の crate を追加してラップすることや、既存モデルごとに座標探索を複製することは解決と認めない。

## 4. 数値計算の是正（検証済み項目）

### 4.1 `betainc_reg` の補集合分枝（P0、即時）

修正は 1 行。補集合側の前置因子は `exp(a·ln x + b·ln(1−x) − ln B) / b` で、**B で割るのは 1 回**。

```rust
// special.rs — 修正後
pub fn betainc_reg(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let log_front = a * x.ln() + b * (-x).ln_1p() - ln_beta;      // ln(1-x) は ln_1p(-x)
    if x < (a + 1.0) / (a + b + 2.0) {
        (log_front.exp() / a) * beta_cf(a, b, x)
    } else {
        1.0 - (log_front.exp() / b) * beta_cf(b, a, 1.0 - x)
    }
}
```

修正版を Python で逐語検証: (a, b) ∈ [0.1, 100]、x ∈ (0,1) の乱数 20,000 点で scipy との最大絶対誤差 **1.9e-11**（残差は 7 項 Lanczos `ln_gamma` と `1e-12` の連分数停止条件由来。1e-13 が要るなら両方を締める）。同梱の `special_oracle_check.py emit golden/special_functions.json` で JSON ゴールデン（1,099 ケース）を生成し、`special.rs` 全関数（`erf`, `norm_cdf`, `ln_gamma`, `digamma`, `gamma_p`, `betainc_reg`, `chi2_cdf`, `student_t_cdf`, `student_t_pvalue`, `f_cdf`, `f_pvalue`）に Tier 0 オラクルテストを付ける。

### 4.2 コサイン冪族の閉形式（171×7 本の手展開を置換）

現行の `cosN_cdf` は N ごとに三角級数を手展開し `1.0 - 1e-15` で clamp している。正しくは正則化不完全ベータ関数 1 本で任意の実数冪に対して計算できる（`betainc_reg` と `ln_gamma` は既に `special.rs` にある）。

記号: 観測 y、位置 μ、尺度 s（推定対象）、冪 n（設計パラメータ）、補助角 θ。

$$\log f{\color{dimgray}{(}}\color{forestgreen}{y}{\color{dimgray}{)}} {\color{dimgray}{=}} \color{coral}{n}\,\log\cos\color{forestgreen}{\theta} {\color{dimgray}{-}} \log \color{dimgray}{Z}_{\color{coral}{n}} {\color{dimgray}{-}} \log\color{royalblue}{s}{\color{dimgray}{,}}\qquad \color{forestgreen}{\theta} {\color{dimgray}{=}} \frac{\pi}{2}\cdot\frac{\color{forestgreen}{y}{\color{dimgray}{-}}\color{royalblue}{\mu}}{\color{royalblue}{s}}{\color{dimgray}{,}}\quad |\color{forestgreen}{\theta}|<\frac{\pi}{2}$$

$$\log \color{dimgray}{Z}_{\color{coral}{n}} {\color{dimgray}{=}} \log 2 {\color{dimgray}{-}} \tfrac{1}{2}\log\pi {\color{dimgray}{+}} \log\Gamma{\color{dimgray}{\bigl(}}\tfrac{\color{coral}{n}+1}{2}{\color{dimgray}{\bigr)}} {\color{dimgray}{-}} \log\Gamma{\color{dimgray}{\bigl(}}\tfrac{\color{coral}{n}}{2}+1{\color{dimgray}{\bigr)}}$$

$$F{\color{dimgray}{(}}\color{forestgreen}{y}{\color{dimgray}{)}} {\color{dimgray}{=}} \tfrac{1}{2}{\color{dimgray}{+}}\tfrac{1}{2}\,\operatorname{sgn}{\color{dimgray}{(}}\color{forestgreen}{\theta}{\color{dimgray}{)}}\; I_{\sin^{2}\color{forestgreen}{\theta}}{\color{dimgray}{\bigl(}}\tfrac{1}{2}{\color{dimgray}{,}}\;\tfrac{\color{coral}{n}+1}{2}{\color{dimgray}{\bigr)}}$$

$$1{\color{dimgray}{-}}F{\color{dimgray}{(}}\color{forestgreen}{y}{\color{dimgray}{)}} {\color{dimgray}{=}} \tfrac{1}{2}\,I_{\cos^{2}\color{forestgreen}{\theta}}{\color{dimgray}{\bigl(}}\tfrac{\color{coral}{n}+1}{2}{\color{dimgray}{,}}\;\tfrac{1}{2}{\color{dimgray}{\bigr)}}\qquad{\color{dimgray}{(}}\color{forestgreen}{\theta}\ge 0{\color{dimgray}{)}}$$

検算: n=3 で `I_{s²}(1/2, 2) = (3/2)s − (1/2)s³` となり現行 `cos3_cdf = 0.5 + 0.75 s − 0.25 s³` に一致。n=0 で Z=2（一様）、n=2 で Z=1。

実装規則:

- `log_prob` は第 1 式を直接計算する。`support` 外は `f64::NEG_INFINITY`。密度を計算して `ln` を取らない。
- 裾（Beta / Kumaraswamy 変換で `ln F`、`ln(1−F)` が要る場合）は第 4 式の補集合恒等式で**引き算せずに**計算する。`(1 - cdf).max(1e-15).ln()` は禁止（R7）。
- 冪 `n` は `PositiveF64` の実行時パラメータ。Cosine と Tsp（two-sided power）はそれぞれ **1 つの** `Emission` 実装。`Unit` / `Beta` / `Kumaraswamy` / `Exponentiated` / `Discrete` は基底分布を包む **1 つの** `Transformed<E>` で表現する（5 変換 × 2 基底 × 173 冪 = 3,460 型 → 3 型）。
- オラクル: 閉形式は `mpmath.quad` による数値積分と `scipy.special.betainc` で JSON ゴールデン化。許容差は実測 × 3〜4（R9）。

### 4.3 forward–backward の underflow（`scaled_forward_backward`）

現行は `alpha[t][j] = acc * log_emit[t][j].exp()` と**対数放射を直接指数化**してからスケーリングしている。全状態で `log_emit[t][·] < −745` になると `exp` が 0 に落ち、`ScaleFactorZero`（「この系列は不可能」）を誤報する。分散が小さいガウス放射に外れ値が 1 点あれば普通に起きる。

修正: 時刻ごとに `m_t = max_j log_emit[t][j]` を引いて指数化し、`log_scale[t] += m_t` として対数尤度に戻す。または全面 log-space（共通 `logsumexp` を使用）。どちらか 1 実装。Viterbi は既に log-space なのでそのまま。

### 4.4 高次差分の誤差増幅（`WindowLag`）

k 次後退差分は二項係数 `C(k, j)` の交代和。`∑|C(k,j)| = 2^k` なので相対誤差はおよそ `2^k · ε`。`Policy::max_difference_order`（既定 8、上限 20）を超える `order` は `IssueCode::NumericalUnderflow` 系ではなく **`Failure`**（設計パラメータの領域外）。`WindowLag { order: NonZeroUsize, window: NonZeroUsize }` の 1 型に統合し、差分は反復で計算する（`C(418,209)` のような係数を作らない）。

### 4.5 数値ポリシーの集約（D8）

`hmm.rs` だけで `1e-15` が 3,840 回。`signlred::Policy` に以下を追加し、本体からリテラルを消す。

| フィールド | 用途 | 既定 |
|---|---|---|
| `log_prob_floor: Option<f64>` | 意図的に対数確率へ下限を置く場合のみ `Some`。適用時は `NumericalCompromise::ProbabilityClamped { original, floor }` を記録 | `None` |
| `underflow_guard: f64` | スケール因子がこれ未満で `ForwardUnderflow` 警告 | `1e-300` |
| `max_difference_order: usize` | §4.4 | 8 |
| `cf_tol: f64`, `cf_max_iter: usize` | 連分数（`betainc_reg`, `gamma_p`）の停止条件 | `1e-15`, 300 |

`NumericalCompromise` に `ProbabilityClamped` / `LogDomainFallback` を追加。`NaN` は `IssueCode::NonFiniteOutput` で表面化させ、置換しない。

---

## 5. HMM 再構成（D9）

```text
kanimiso/src/hmm/
├── mod.rs                  # pub use と最小ファサード
├── model.rs                # HiddenMarkovModel<E> { initial, transition, emissions: Vec<E> }
├── forward_backward.rs     # 1 実装（§4.3 の修正込み）
├── viterbi.rs
├── baum_welch.rs           # Emission::{accumulate, maximize} を呼ぶ汎用 EM
├── diagnostics.rs          # AbsorbingStateOnly / UnreachableState / EmissionDegenerate の検出
└── emission/
    ├── mod.rs              # trait Emission
    ├── gaussian.rs
    ├── poisson.rs
    ├── categorical.rs
    ├── cosine_power.rs     # §4.2、冪は実行時
    ├── two_sided_power.rs
    └── transformed.rs      # Transformed<E> { base: E, transform: Unit|Beta|Kumaraswamy|Exponentiated|Discrete }
```

```rust
pub trait Emission {
    type Observation;
    type SufficientStats: Default;
    fn log_prob(&self, obs: &Self::Observation) -> f64;               // support 外は -∞、NaN 禁止
    fn accumulate(&self, obs: &Self::Observation, weight: f64, stats: &mut Self::SufficientStats);
    fn maximize(&mut self, stats: &Self::SufficientStats, ctx: &mut FitCtx) -> Result<()>;
}
```

- `fit` は `&self` で `Qualified<Fitted>` を返し、状態変更は `PartialFit` だけ（`traits.rs` の `Fit::fit(&mut self, …)` を改める）。`Qualified<T>` に `#[must_use]`。
- **残す分布族の基準**: (a) scipy.stats か閉形式のオラクルがある、(b) `Emission` として 1 ファイルに収まる、(c) 利用理由を issue に書ける。三条件を満たさない生成族は削除し、名前だけを `docs/dropped_v0_1.md` に残す。1,476 族を全部移植する計画は立てない。
- 完了条件: `hmm/` 合計 **2 万行以下**、各ファイル 3,000 行以下、Gaussian / Poisson / Categorical が hmmlearn ゴールデンと一致、brute-force 列挙（T≤6, K≤3）と forward の対数尤度一致、状態置換不変性、EM 単調性。

---

## 6. テスト戦略

| 層 | 内容 | 例 |
|---|---|---|
| Tier 0: オラクル（差分） | scipy / statsmodels / hmmlearn / river / POT で作った **JSON ゴールデン**をリプレイ。Python は実行時依存にしない（`golden/` にコミット、生成スクリプトは PEP 723 で `uv run`） | `special_oracle_check.py emit golden/special_functions.json` |
| Tier 1: 閉形式・brute-force | 小問題での全列挙、解析解 | HMM 尤度 vs 全経路列挙、Viterbi vs argmax 全列挙 |
| Tier 2: プロパティ | CDF 単調・[0,1]、確率和 1、有限性、状態置換不変、`Emission::log_prob` の support 外 `-∞`、遷移行列行和 1、EM で尤度が許容差以上に悪化しない | 固定 seed の決定的生成。専用 property-test crate は追加しない |
| Tier 3: ストレス | 特異行列、極端値、長系列（T=1e5）、欠損、定数系列、`log_emit` が全て −1e4 の系列（§4.3） | 落ちない・`NaN` を返さない・`Issue` を残す |

- **許容差の決め方（R9）**: まず実装と実測誤差を記録し、`×3〜4` を閾値にする。テストにはコメントで実測値を残す（例: `// measured 2.1e-12 on 2026-09-01, tol = 1e-11`）。
- 生成型ごとの smoke test（`gaussian_hmm_learns_two_means` を 171 通りコピーする類）は書かない。
- 外部参照境界: 参照実装から生成した決定的 golden と共通核の `value` が一致し、`report` が空でない場合は理由があることを検査する。外部 crate はテスト時にも依存へ追加しない。

---

## 7. CI（PR 1 で導入、以後必須チェック）

```text
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features            # RUST_MIN_STACK なし
cargo test --workspace --doc
cargo doc --workspace --no-deps  (RUSTDOCFLAGS="-D warnings")
cargo llvm-cov --workspace --fail-under-lines <ratchet>
cargo deny check
scripts/lint_redundancy.sh                        # 付録 A: R1 / R4 / R5 / R7 / R10、RUST_MIN_STACK 不在、ndarray/faer 単一版
```

- マトリクス: `stable` と `1.85`（workspace MSRV）。1.85 で落ちたら D7 に従って全 workspace の `rust-version` を同時に上げる。
- `main` への直接 push を禁止し、全チェックを required にする。
- PR テンプレート必須欄: 「削減した行数 / pub 項目数（前 → 後）」「オラクル: 何と、どの許容差で、実測値」「同型の既存実装が無いことの確認」。

---

## 8. PR 順序と完了条件

| PR | 内容 | 完了条件 |
|---|---|---|
| 1 | タグ `generated-v0.1-archive`、CI（§7）、`scripts/lint_redundancy.sh`（付録 A、予算は実測値で開始）、`coverage.rs` に `status`、README の主張縮小。`RUST_MIN_STACK` は PR 8 まで残す（`ALLOW_RUST_MIN_STACK=1`） | `main` に required check。lint が現状コードで通る |
| 2 | `special.rs` の scipy ゴールデン + **`betainc_reg` 修正**（§4.1）+ `cf_tol` を `Policy` へ | `special_functions.json` 全件 pass、p 値が [0,1] を出るテスト追加 |
| 3 | `ndarray = 0.17.2` / `faer = 0.24.4` の依存境界（§3）、lock / feature / native backend lint | 両 crate が単一版、1.85 ジョブ pass、native BLAS/LAPACK 不在 |
| 4 | `WindowLag1..418` → 1 型（§4.4） | 型 1 つ、`order ≤ 8` の代表ケースで v0.1 と一致、`order > max` は `Failure` |
| 5 | `LogMinkowski1..356Anomaly` → `KnnDistanceAnomaly` | 距離核は workspace 内で 1 実装、`p` 実行時、代表 `p ∈ {1, 2, 5, ∞}` で一致 |
| 6 | `Emission` trait + `hmm/` core（§5）、Gaussian / Poisson / Categorical | hmmlearn ゴールデン一致、brute-force 一致、置換不変、EM 単調、§4.3 のストレス系列で `NaN` なし |
| 7 | `cosine_power` / `two_sided_power` / `Transformed<E>`（§4.2） | mpmath ゴールデン一致、n=3 で旧 `cos3_cdf` と一致、裾で `ln(1−F)` が有限 |
| 8 | 生成 HMM 型の全削除、`hmm.rs` 削除、`RUST_MIN_STACK` 削除 | `hmm/` 2 万行以下、環境変数なしで `cargo test` pass |
| 9 | DTW / kernel / 距離のモデル別コピーを workspace の単一共通核へ統合 | 同じ DP・カーネル式・距離式が複数モジュールに残っていない（grep で確認） |
| 10 | プロパティ / 差分テスト整備（§6）、`llvm-cov` の ratchet 設定 | 数値カーネル全てに Tier 0 か 1 + Tier 2 |
| 11 | 公開 API 整理、`Fit::fit(&self)`、`#[must_use] Qualified` | `pub` 1,000 以下、`cargo public-api` の差分レビュー済み |
| 12 | `0.2.0-alpha.1` | docs / examples / CI 全通過、`coverage` の `verified` と README が一致 |

---

## 9. 目標値

| 指標 | 現在 | alpha.1 |
|---|---:|---:|
| Rust 総行数 | 964,218 | 100,000〜150,000 |
| 最大ファイル | 568,846 行 | 3,000 行 |
| `pub` 項目（kanimiso） | 19,739 | 500〜1,000 |
| HMM 型数 | 4,956 | 10 前後（`HiddenMarkovModel<E>` + 放射 6〜8 + `Transformed`） |
| テスト | 186 smoke | 全数値カーネルに Tier 0/1 + Tier 2 |
| `RUST_MIN_STACK` | 32 MiB | なし |
| CI required checks | 0 | §7 全部 + MSRV マトリクス |
| ndarray / faer 版 | 0.17.2 / 0.24.4 | 各 1 版を CI 保証 |

---

## 10. エージェント作業規則

1. **読まずに書かない**。変更対象と、同じ概念の既存実装（R2, R6）を先に探す。
2. **計測してから、計測して終わる**。PR 説明に行数と `pub` 項目数の前後を書く。増えていたら理由を書く。
3. **1 PR 1 概念**。リファクタ PR に新アルゴリズムを混ぜない（D1）。
4. **v0.1 と結果が変わったら理由を書く**。旧値との不一致は不合格条件ではない。オラクルとの不一致が不合格条件（D2）。
5. **数値リテラルを書いたら手を止める**（R7）。`Policy` に無ければ追加してから参照する。
6. **`Foo2` / マクロ生成 / `Distinct from` を書きたくなったら止まる**（R1, R3, R10）。
7. **共通数値核を再実装しない**。外部実装は参照・オラクルに使い、依存へ追加しない（D4, R6, §3）。
8. **`cargo test` が環境変数なしで通るまで「動いた」と言わない**。
9. **コンテキストが足りなくなったら区切る**。中途半端な削除は最悪（生成型の半分だけ消えた状態はビルドもテストも壊れる）。区切るときは現状と次の一手を PR 説明に書く。

---

## 付録 A: `scripts/lint_redundancy.sh`

全項目が **ratchet 予算**。初期値は 2026-09-01 の実測値なので PR 1 の時点で通る。以後、予算は**下げる方向にしか変更できず**、実測が予算を超えた PR は落ちる。R1 / R7 / R10 の最終目標は 0。

```bash
#!/usr/bin/env bash
# 冗長性 lint（R1 / R4 / R5 / R7 / R10）+ スタック拡張 + ndarray/faer 単一版。全て ratchet 予算。
set -uo pipefail
SRC="kanimiso/src signlred/src ojizou-san/src"
ALLOW=scripts/lint_allowlist.txt        # 固有名詞の数値（Exp3, Ucb1, Catch22, X13, 2Sls, Chi2, F1, R2 …）を 1 行 1 固定文字列
[ -f "$ALLOW" ] || : > "$ALLOW"

# ---- 予算（下げるのみ。初期値 = 2026-09-01 実測）-----------------------------
MAX_R1_NUMERAL_IDENTS=4437     # 目標 0  （Garch11 のような「パラメータ値」は allowlist に入れない）
MAX_R4_PUB_ITEMS=18195         # 目標 1000
MAX_R5_FILES_OVER_3000=13      # 目標 0
MAX_R7_DENSITY_FLOORS=1135     # 目標 0
MAX_R10_DISTINCT_FROM=3021     # 目標 0
ALLOW_RUST_MIN_STACK=1         # PR 8 で 0 にする
# ------------------------------------------------------------------------------
fail=0
budget() { # name actual max
  if [ "$2" -gt "$3" ]; then echo "FAIL $1: $2 > budget $3"; fail=1; else echo "ok   $1: $2 (budget $3)"; fi
}

python3 scripts/lint_dependencies.py || fail=1

r1=$(grep -rhE '^\s*pub (struct|enum|trait|fn|type|mod) [A-Za-z_]*[0-9]+[A-Za-z_0-9]*' --include='*.rs' $SRC \
     | grep -vF -f "$ALLOW" | wc -l)
budget "R1 numeral in public identifier" "$r1" "$MAX_R1_NUMERAL_IDENTS"

r4=$(grep -rhoE '^\s*pub (struct|enum|trait|fn|type|const|mod|static) ' --include='*.rs' kanimiso/src | wc -l)
budget "R4 pub items (kanimiso)" "$r4" "$MAX_R4_PUB_ITEMS"

r5=$(find $SRC -name '*.rs' | xargs wc -l | grep -v ' total$' | awk '$1>3000' | wc -l)
budget "R5 files over 3000 lines" "$r5" "$MAX_R5_FILES_OVER_3000"

r7=$(grep -rhE '\.max\(1e-[0-9]+\)\.ln\(\)|\.clamp\(0\.0, 1\.0 - 1e' --include='*.rs' $SRC | wc -l)
budget "R7 density floor / probability clamp" "$r7" "$MAX_R7_DENSITY_FLOORS"

r10=$(grep -rh 'Distinct from \[`' --include='*.rs' $SRC | wc -l)
budget "R10 'Distinct from' docstrings" "$r10" "$MAX_R10_DISTINCT_FROM"

if [ "$ALLOW_RUST_MIN_STACK" -eq 0 ] && grep -q RUST_MIN_STACK .cargo/config.toml 2>/dev/null; then
  echo "FAIL RUST_MIN_STACK is forbidden"; fail=1; fi

faer_versions=$(cargo tree -i faer -e normal --prefix none 2>/dev/null | awk '/^faer /' | sort -u | wc -l)
if [ "$faer_versions" -gt 1 ]; then echo "FAIL multiple faer versions in the graph"; fail=1; fi

ndarray_versions=$(cargo tree -i ndarray -e normal --prefix none 2>/dev/null | awk '/^ndarray /' | sort -u | wc -l)
if [ "$ndarray_versions" -gt 1 ]; then echo "FAIL multiple ndarray versions in the graph"; fail=1; fi

exit $fail
```

運用規則:

- 予算を**上げる**変更を含む PR は差し戻し。予算を下げる変更は削減 PR に同梱する。
- `lint_allowlist.txt` に入れられるのは**固有名詞としての数値**（アルゴリズム名・統計量名）のみ。`Garch11`（p=1, q=1 のパラメータ値）や `WindowLag7` は入れない。追加時は PR 説明で正当化する。
- 実測を予算にする R9 と同じ思想: 閾値の由来が追える。

## 付録 B: 未決事項（やる夫の判断待ち）

| # | 論点 | 既定値（判断がなければこれで進む） |
|---|---|---|
| B1 | 1,476 HMM 分布族のうち v0.2 に残すもの | §5 の三条件を満たすものだけ。残りは削除し `docs/dropped_v0_1.md` に名前のみ |
| B2 | ndarray と faer の adapter を公開 API にするか | 必要な利用例が現れるまで増やさない。追加時はコピーを名前で明示し 1 モジュールに限定 |
| B3 | toolchain 1.98 に対して MSRV 1.85 を維持できるか | workspace 全体を CI の Rust 1.85 で実ビルドし、通らない場合だけ全 crate 同時に上げる |
| B4 | `NumericalPolicy` を `signlred::Policy` 拡張で持つか、`kanimiso` 側に別置きするか | `signlred::Policy` 拡張（D8） |
| B5 | `special.rs` を kanimiso に残すか、独立 workspace crate にするか | v0.2 では kanimiso 内に残す。複数 first-party crate が必要とした時点でだけ切り出す |
| B6 | 生成型削除の互換性告知 | 安定版利用者がいないため告知なし。`CHANGELOG.md` に「v0.1 の生成型は全削除」と 1 行 |
