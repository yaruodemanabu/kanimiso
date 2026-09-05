# kanimiso 残ソースのサブクレート化と `ogi` 計画（2026-09-05）

実装はしない。切り出しの境界、順序、検証契約、未決事項を固定するための計画である。
現状の実測と穴は [`spec_and_test_coverage.md`](spec_and_test_coverage.md) に書いた。
この文書はその続きで、「残っている `kanimiso/src` をどこへ移すか」だけを扱う。

方針の前提（AGENTS.md）:

- 1 PR 1 概念。切り出し PR に新アルゴリズムを混ぜない（D1）
- 数値核は増やさない。`tsutsumi` / `signlred` / `ojizou-san` を呼ぶ（R6）
- 中途半端な削除はしない。移した単位でビルドと再生テストが通るまで「動いた」と言わない
- v0.1 との数値一致は受入条件ではない。オラクルと台帳が受入条件（D2, D10）

---

## 1. 目標

`kanimiso` を **統合ファサード** にする。推定核はすべて first-party サブクレートに置き、
`kanimiso` は再公開・品質アダプター・台帳だけを持つ。

すでに切り出済み:

| crate | 核 |
|---|---|
| `tsutsumi` | 行列、分解、特殊関数、Nelder–Mead、求積、`FitCtx` / traits |
| `number-ruler` | 注釈付き LM / GLM / LMM / GLMM / LAM / GAM / 線形 SHAP |
| `oldwood` / `mayoi-no-mori` | CART / ensemble |
| `denshi` | Hedge と射影勾配（**リグレット**のオンライン予測） |
| `riko` | UCB と指数重みバンディット（**リグレット**） |
| `signlred` / `ojizou-san` | 品質契約と ledger |

残る `kanimiso/src` のうち、リグレット予測ではない

- 時系列（離散時間の計量・ボラティリティ・状態空間・HMM・共和分）
- 逐次推定（river 風のオンライン統計。損失ベクトルを見て意思決定しない）
- ベイズ（リッジ / ARD / GP など、バッチまたは逐次の事後・証拠最大化）

を **`ogi` に集約する**。これが指定された次の切り出し先である。

`kanimiso` に残る sklearn 風の巨大面（`glm` `cluster` `metrics` …）は `ogi` ではない。
別クレートへ移すか、検証計画付きで残すか、アーカイブする。名前は未決（§8）。

---

## 2. 完成形の依存

```text
signlred, ojizou-san
        │
        ▼
     tsutsumi          amatsuki (ChaCha8, core-only)
        │                    │
        ├──────────┬─────────┤
        ▼          ▼         ▼
      ogi    number-ruler   oldwood
        │          │         │
        │          │         ▼
        │          │    mayoi-no-mori
        │          │
        ▼          ▼
   denshi / riko     （ogi にも mayoi にも依存しない）
        │
        ▼
     kanimiso   = 再公開 + coverage 台帳 + tree/histgb アダプター
```

`ogi` が依存してよいのは `tsutsumi`（`linalg` feature）、`signlred`、`ojizou-san` だけである。
`number-ruler`、`oldwood`、`mayoi-no-mori`、`denshi`、`riko`、`Isuzu`、`wormhole` には依存しない。
乱数が要る公開 API（いまは `arma_generate_sample`）は、切り出し時に

1. 呼び出し側が乱数を渡す（`riko` 方式）、または
2. first-party `amatsuki` の core-only、または
3. 既存の xorshift を `ogi` 内に閉じる

のどれかにする。`kanimiso::rng` への逆依存は作らない（§8-Q5）。

公開順の追記案:

`signlred` → `ojizou-san` → `tsutsumi` → `number-ruler` / `oldwood` / **`ogi`** →
`denshi` / `riko` → `mayoi-no-mori` → `kanimiso`。

---

## 3. `ogi` に入れるもの / 入れないもの

### 3.1 第 1 波（台帳あり。先に移す）

いまの検証契約の本体。行数は 2026-09-05 実測。

| いまの場所 | 行 / `#[test]` | 台帳 | golden / CI | `ogi` での置き場 |
|---|---:|---|---|---|
| `online/` | 4,682 / 25 | Verified 11 | `online_rls.json` + モーメント閉形式。`online-rls-oracle` | `ogi::online` |
| `state_space/` | 1,678 / 18 | Verified 1 | `state_space.json`。`state-space-oracle` | `ogi::state_space` |
| `filters.rs` | 1,243 / 11 | Verified 7 | statsmodels ソース + 閉形式。`time-series-oracles` | `ogi::filters` |
| `tsa/` | 9,136 / 82 | Verified 8 + Experimental 3 | 各 `*_qml.json` / `arma_acov.json`。`garch-oracle` / `time-series-oracles` | `ogi::tsa` |
| `stats/` | 580 / 3 | Verified 1 | `process_mle.json` | `ogi::stats` |
| `hmm/` | 3,022 / 33 | Experimental 4 | `hmm.json`。`hmm-core` | `ogi::hmm` |

合計 roughly **20,300 行、172 テスト**。HMM 完了条件の「`hmm/` 2 万行以下」は
`ogi` 全体でもこの波では満たす。個別ファイルはいま 3,000 行超えていない
（`tsa/figarch.rs` が 2,398 で最大級）。

`tsa` は `filters` を再公開している。`state_space` ← `filters` ← `tsa` の順で
同じ crate に入れる。分割するなら `state_space`+`filters` を先、`tsa` を次。

`anomaly.KnnDistanceAnomaly` は台帳 Experimental だが、`online` の内部 helper に加え
`tree` / `svm` / `neighbors` を呼ぶ。**第 1 波には入れない。** helper は
`ogi::online` から公開するか、anomaly 側に複製せず残置アダプタで呼ぶ（§8-Q6）。

### 3.2 第 2 波（時系列・ベイズだが未登録）

テーマは `ogi` だが、台帳にも独立 golden もない。移す前に
Experimental 登録か、`dropped_v0_1` 行きかを決める。

| いまの場所 | 行 / テスト | 中身 | 暫定判断 |
|---|---:|---|---|
| `vecm.rs` | 579 / 2 | Johansen / VECM | `ogi` 候補。オラクル無し。移すなら Experimental |
| `reducer.rs` | 598 / 2 | sktime 風のラグ還元予測 | 同上 |
| `bayes.rs` | 429 / 2 | BayesianRidge, ARD | `ogi` 候補（ベイズ線形） |
| `gp.rs` | 459 / 2 | RBF GP 回帰 / Laplace 分類 | `ogi` 候補 |
| `topic.rs` | 297 / 1 | バッチ VB の LDA | 境界。テキストモデルであり計量時系列ではない（§8-Q3） |
| `naive_bayes.rs` | 1,230 / 6 | sklearn 風 NB + `PartialFit` | 境界。分類器であり、逐次推定ではあるがベイズ時系列ではない（§8-Q2） |

第 2 波を第 1 波と同じ PR に載せない。未登録面を `ogi` に入れると、
crate README の Verified 主張が薄まる。

### 3.3 `ogi` に入れない（明確）

| 面 | 理由 | 行き先 |
|---|---|---|
| `denshi` / `riko` | リグレット / バンディット。指定どおり `ogi` の外 | 現状維持。台帳・README への掲載は別判断 |
| `kanimiso::bandit` | 同じリグレット族。`riko` と重複 | `riko` へ寄せるか削除。`ogi` ではない |
| `number-ruler` と `kanimiso::mixed` | 混合モデルは回帰注釈の側 | 重複を潰して `number-ruler` かアーカイブ |
| `kanimiso::glm` / `linear_model` の未登録推定器 | 時系列でもベイズでもない | 未決の supervised 束（§5） |
| `tree` / `histgb` | 既に `oldwood` / `mayoi-no-mori` | ファサードに残す |
| Isuzu の SDE / 粒子フィルタ / 金融 / npbayes | 連続時間・点過程の別スタック | workspace 外のまま。Kalman 核の統合は後（§8-Q4） |
| wormhole / coronel / jelly-wave | 輸送・カーネル・DTW | ルート検証契約の外。PR 9 の距離統合と別件 |
| AGENTS.md §5 の `cosine_power` / `TSP` / `Transformed` | 未実装 | 実装するなら最初から `ogi::hmm`（kanimiso に足さない） |

`WindowLag` を復活させるなら `ogi::online` であり `denshi` ではない
（差分は逐次統計。リグレットではない）。いまはアーカイブのまま。

---

## 4. `ogi` の公開面（第 1 波）

crate ルートは小さくする。モジュール名に数値パラメータを埋め込まない（R1）。
`Garch11` は既存の固有名詞として allowlist 済みなら維持。新規に `Garch11` を増やさない。

```text
ogi/
  Cargo.toml
  README.md
  docs/validation.md          # ルート validation から時系列節を移す
  golden/                     # 第 1 波の fixture をここに集約するか、ルート golden を参照（§8-Q8）
  src/
    lib.rs
    online/
    state_space/
    filters.rs
    tsa/
    stats/
    hmm/
```

`kanimiso` 側は移したあと

```rust
pub use ogi::online;
pub use ogi::state_space;
pub use ogi::filters;
pub use ogi::tsa;
pub use ogi::stats;
pub use ogi::hmm;
```

のように再公開する。利用者パス `kanimiso::online::LinearRegression` を
いきなり消さない。台帳名（`online.*`, `tsa.*`）も当面そのまま。
`ogi::` を正本にする改名は、切り出しが全部終わってから別 PR（§8-Q7）。

`Fit` / `FitSeries` / `PartialFit` は `tsutsumi::traits` を使い続ける。
`ogi` に第二の trait を置かない。

---

## 5. `ogi` 以外の残ソース（名前は未決）

`kanimiso` をファサードだけにするには、`ogi` のあと（または並行して）
次の束を決める必要がある。ここでは **crate 名を付けない。** 付けると
切り出しが先行してしまう。

| 束 | いまのモジュール（行数概算） | 既存クレートとの関係 |
|---|---|---|
| リグレット残り | `bandit` 1,720 | `riko` に吸収 or 削除 |
| 回帰の重複 | `mixed` 825、`linear_model` の OLS 以外、`glm` 6,185、`robust` 1,391 | `number-ruler` の範囲を超えるものはアーカイブ候補が強い |
| 木アダプタ | `tree` 1,518、`histgb` 362、`ensemble` の AdaBoost 再公開 | ファサードに残してよい |
| sklearn 風教師あり / 教師なし | `cluster` `svm` `neural` `decompose` `metrics` `feature` `preprocess` `neighbors` `classification` `covariance` `discriminant` `compose` `model_selection` `multiclass` `multinomial` `multioutput` `manifold` `kernel_pca` `random_projection` `text` `semi` など、未登録側合計 ~5 万行 | 検証計画が無い新規核は D1 違反。束ごとアーカイブするか、オラクル付きで 1 概念ずつ出す |
| 乱数 | `rng` 415 | `amatsuki` に寄せるか、決定的生成だけ `ogi` に閉じる |
| 異常検知ラッパ | `anomaly` 1,049 | 核（k-NN）とラッパを分け、ラッパはファサード |

R5 超過 6 ファイルはすべてこの「`ogi` 以外」側にある。
`ogi` 切り出しだけでは R5 は下がらない。予算を下げる作業は別 PR。

---

## 6. 検証・CI・台帳の移し方

切り出しは **テストと fixture を核と一緒に移す。** 空の crate を先に作って
中身を後から、はしない。

1. `ogi/docs/validation.md` を新設し、ルート `docs/validation.md` の
   online / 状態空間 / ARMA / フィルタ / ボラティリティ / process MLE / HMM を移す。
   ルート文書は「`ogi` が正本、ここは要約」にする。
2. CI の再生コマンドを `-p kanimiso --lib online::…` から `-p ogi --lib …` に変える。
   Python `scripts/*_oracle.py check` は残してよい。
3. `quality-small` の clippy / rustdoc 対象に `ogi` を足す。
   `denshi` / `riko` を同時に足すかは別判断（カバー文書 §8-3）。
4. `coverage.rs` の `registered_paths_link_to_active_symbols` は
   `kanimiso::` 再公開経由のまま通る。名前の書き換えは改名 PR までしない。
5. `cargo package -p ogi --list` を workspace-policy に足す。
6. line coverage は workspace 合計のまま。核が `kanimiso` から減るので、
   未登録モジュールの比重が上がる。ratchet を「核の切り出しで下がった」と
   誤読しないこと（カバー文書 §8-11）。必要なら切り出し PR で実測し、
   **下げる方向だけ** 予算を更新する。上げない。

オラクルを新しく作る作業は切り出しに混ぜない。
EWMA / EGARCH / HMM / Nelder–Mead の昇格も混ぜない。

---

## 7. PR 列（1 概念ずつ）

各 PR の完了条件: 移したテストが同じ fixture で通る、`kanimiso` の再公開が
コンパイルされる、R4 が上がらない、新しい `pub` を増やした理由を書く。

| 順 | 内容 | 完了の測り方 |
|---|---|---|
| O1 | `ogi` crate 骨格（README、validation 空でない、workspace member、依存は上記のみ） | `cargo test -p ogi` が空でも通る。アルゴリズムはまだ移さないか、次と同一 PR にしない |
| O2 | Verified `online/` を移動 + `kanimiso::online` 再公開 | `online-rls-oracle` 相当を `-p ogi` で再生。anomaly はまだ `kanimiso` |
| O3 | `state_space/` と `filters.rs` を移動 | `state-space-oracle` とフィルタテストを `-p ogi` |
| O4 | `tsa/` と `stats/process` を移動 | ARMA / process / GARCH 族ジョブを `-p ogi`。`arma_generate_sample` の RNG 方針をこの PR で決めて実装 |
| O5 | `hmm/` を移動 | `hmm-core` を `-p ogi`。cosine 族はまだ書かない |
| O6 | ルート validation / README ワークスペース表 / CHANGELOG / 公開順 | 文書だけ。コード移動なし |
| O7 | 第 2 波のうち **残すと決めたものだけ** を Experimental で移動 | 台帳に名前を足す。golden が無いものは Verified にしない |
| O8 | `kanimiso::bandit` を `riko` へ寄せるか削除 | `ogi` ではない。O1–O5 の後 |
| O9 | 残 supervised 束の行き先決定（アーカイブ or 新 crate） | 決定書。大量移動は決定の次 |
| O10 | `kanimiso` から実装ファイルを消し、ファサード + 台帳だけにする | `kanimiso/src` が再公開とアダプタと `coverage.rs` に近い状態 |

O1 と O2 を一つの PR にまとめてよい（空 crate だけの PR は意味が薄い）。
O3 と O4 は依存があるので連続にするが、差分が大きくなるなら分けたまま。
O7 を O2 に混ぜない。

AGENTS.md §8 の既存列との関係:

- PR 7（cosine / TSP / `Transformed`）は、O5 のあとなら最初から `ogi::hmm` に書く
- PR 9（距離核の単一化）は `ogi` と独立。`kanimiso::metrics` と wormhole の重複調査が先
- PR 11（`pub` 1,000 以下）は、未登録面をアーカイブしないと届かない。O9 が前提

---

## 8. 疑問点（切り出し前に決めたい）

カバー文書 §8 と重なるものは番号を共有しない。こちらは切り出し固有。

1. **`ogi` の表示名と一文の責務。** コード名は指定どおり `ogi`。README 先頭の
   日本語一文（何の crate で、何ではないか）をどう書くか。
2. **`naive_bayes` は `ogi` か。** 逐次更新はあるが、計量時系列でも回帰ベイズでもない。
   supervised 束に残す方が境界がきれい、という案を既定にする。
3. **`topic`（LDA）は `ogi` か。** VB という点ではベイズ。テキストモデルという点では外。
   既定は外（第 2 波にも入れない）。
4. **Isuzu の Kalman / QMLE を将来 `ogi` の核へ寄せるか。** 今は寄せない。
   寄せると deviations とベンチが壊れ、D4（`rustfft` 等）も絡む。
5. **`arma_generate_sample` の RNG。** 呼び出し側供給 / `amatsuki` / 現行 xorshift のどれか。
   `riko` は隠れ RNG を拒否している。`ogi` を同じ契約にするか。
6. **`online` の `pub(crate)` helper を `ogi` でどう公開するか。**
   `anomaly` が使う。`pub` に上げると R4 が増える。`ogi` 内 `pub(crate)` のまま
   anomaly を後で移すまで `kanimiso` 側に薄い橋を残す、が既定案。
7. **台帳パスを `ogi.online.LinearRegression` に変えるか。** 当面は変えない既定。
8. **golden を `ogi/golden/` に移すか。** 移すと Python スクリプトと CI のパスが変わる。
   第 1 波はルート `golden/` を `include_str!` で読む方が差分が小さい。
9. **`optimize.NelderMead` は `tsutsumi` のままか。** 既定はまま。`ogi` は呼ばない核を持たない。
10. **第 2 波を Experimental で残すか、先にアーカイブするか。**
    `vecm` / `reducer` / `bayes` / `gp` に検証計画が無いなら D1 的にはアーカイブが先。
11. **supervised 残り束の crate 名。** この計画では付けない。先に「残す集合」を決める。
12. **`kanimiso` が `ogi` を再公開したあと、利用者はどちらを依存に書くか。**
    既定案: 核だけ欲しい人は `ogi`、木や回帰も要る人は `kanimiso`。
13. **`LocalLinearTrend` は `filters` に置いたままか、`state_space` に寄せるか。**
    移動のついでにモジュールを変えると 1 PR 2 概念になる。既定はパス維持。
14. **`Garch11` という公開名を `Garch { p, q }` に変えるか。** R1 の本丸だが、
    切り出しと同時にやらない。allowlist 維持で移す。

---

## 9. やらないこと（この計画の範囲外）

- 新モデル、cosine 冪、hmmlearn golden、ensemble 外部 fixture
- Isuzu / wormhole の workspace 編入
- `denshi` / `riko` の台帳掲載（別判断）
- 未登録 5 万行の一括削除
- 方針文書 AGENTS.md §0 の数値更新（別 PR でよい）

---

## 10. 最初の実装 PR が始まる条件

次が揃ったら O1+O2 に進んでよい。

- §8 の Q2, Q3, Q5, Q10 について、この文書の既定を採用するか否かの一言
- `ogi` の README 一文（Q1）
- golden はルートに残す（Q8 の既定）でよいか

コードを書き始める PR では、この計画の「第 1 波 online だけ」を超えない。
