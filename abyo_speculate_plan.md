# abyo-speculate 企画書

ローカル LLM 向け Speculative Decoding 統合ライブラリ (Pure Rust)

- 作成日: 2026-05-09
- 著者: abyo software, LLC
- ステータス: 企画 (未着手)
- 関連: abyo-llm-probe (先行プロジェクト、Sprint 3 で v0.1 公開予定)

---

## エグゼクティブサマリ

**何を作るか**: Medusa / EAGLE-2 / EAGLE-3 / SAGUARO / Recurrent Drafter 等の Speculative Decoding (SD) 系手法を統一 API で提供する Pure Rust ライブラリ。

**誰のために**: ローカル LLM ユーザ (Ollama / LM Studio / llama.cpp / candle 利用者、Rust エージェント開発者)。バッチサイズ 1 の単一ユーザ運用に特化。

**差別化軸**: 「バッチサイズ 1 のローカル運用 × 主要オープンモデル (Llama 3.x / Qwen 2.5 / Mistral / Phi-3.5) 特化 × 統一 API で複数手法切り替え」。

**狙い**: 直接収益はゼロ前提。**Ferro 売却バリュエーション向上のブランディング投資**として、abyo-llm-probe との二段ロケットで「ローカル LLM 推論基盤の専門家」ポジションを確立する。

**主要リスク**:
1. llama.cpp の EAGLE-3 PR (#18039) 完成で価値半減 → 新手法追従が必須
2. MoE / 高温度サンプリング / 短文生成では速くならない → 正直に明記する
3. Draft head の調達 (各 target model 用に必要) → ランタイムだけに絞るか学習サービスまで含むかの判断要

**工数**: Sprint S1〜S3 で 15〜21 日 (probe ノウハウで 12〜18 日に圧縮可能)。

---

## 1. なぜ今これをやるか

統合ロードマップで棚上げになっていた候補を、abyo-llm-probe の実装が動き始めた前提で再評価した結果。

### 採択理由

1. **市場が大きい**: ローカル LLM ユーザは爆発的に成長中。推論速度が GPU/VRAM コストに直結する。
2. **数字で殴れる**: Medusa 2.3×、EAGLE-2/3 3.5×、SAGUARO 5× (autoregressive 比、論文値)。HN タイトルとして強い。
3. **probe との棲み分け**: probe = ローカル LLM をエンコーダ化、speculate = 生成器の高速化。**同じユーザに 2 ライブラリ売れる**。
4. **probe の経験が活きる**: candle / mistral.rs 操作、KV キャッシュ、forward hook 等の実装ノウハウが流用できる。
5. **Rust 統合実装の空白**: vLLM / SGLang は Python、ローカル界隈は llama.cpp のみ。**統合 Rust ライブラリは皆無**。

### 他候補を選ばなかった理由

| 候補 | 不採択理由 |
|------|------|
| abyo-memorable | 工数軽い (5〜7 日) が市場サイズが abyo-speculate より一桁小さい |
| abyo-memory | Mem0 が YC 通過済で先行、差別化軸が弱い |
| abyo-activation-memory | KV Cache + 意味検索は RAG で代替可能、需要不明確 |

---

## 2. 推奨ポジション (結論)

「**バッチサイズ 1 のローカル運用、Llama 3.x / Qwen 2.5 / Mistral / Phi-3.5 に絞った最強実装**」

これにより:

- ターゲットが明確 (個人開発者、Rust エージェント運用者)
- 「全モデル全タスクで速い」という詐欺的主張を回避できる
- vLLM / SGLang と棲み分け (彼らはデータセンター・バッチ処理向け)
- llama.cpp と棲み分け (彼らは C++、Rust 界隈は空白)

### 棄却した方向性

- **方向 C: 全手法網羅** — 実装範囲が広すぎる。論文ポートフォリオとしては魅力的だが工数破綻。
- **方向 N: 全モデル対応** — vLLM の劣化コピーになる。

### 採用 (推奨方向 A + B のハイブリッド) の弱み

- 単独の「方向 A: バッチ 1 特化」だけだと市場サイズが薄い → モデル特化 (B) で広げる
- モデル追加のメンテナンスコストは事実発生する → 4 モデルに固定して許容

---

## 3. 正直に書く弱点・リスク

宣伝資料じゃないので、実装着手前に知っておくべき難所を全部書く。

### 3.1 既存実装が成熟してきている

2025-2026 で speculative decoding は **「研究実験」から「本番標準」へ移行**した:

- vLLM、SGLang、TensorRT-LLM はネイティブ対応済み
- llama.cpp も EAGLE-3 PR が進行中 (#18039、2026-04 時点で draft)
- NVIDIA は H200 で 3.6× スループット改善を実証済み

つまり「Rust で書きました」だけでは差別化にならない。**「既存実装が拾えていない領域」**を明確に取りに行く必要がある。本企画ではそれを「バッチ 1 × 主要 4 モデル × 手法切り替え」とした。

### 3.2 コンシューマー GPU で効果が薄いケース

dev.to の実験 (2026-04) で報告されている事実:

- **n-gram speculative decoding は家庭用 GPU クラスタで効果なし**
- **MoE モデルで 0.89〜1.06×** (一部のプロンプトでむしろ遅い)
- 専門家活性化のオーバーヘッドが verification を遅くする

これは「ローカル LLM 高速化」を売りにする時の最大のリスク。**「全モデル全タスクで速くなる」と主張すると詐欺になる**。

### 3.3 バッチサイズが大きいと効果が落ちる

EAGLE 系の研究 (Li et al., 2025) が実証している:

- バッチサイズ 2: スピードアップ大
- バッチサイズ 48: 速度向上が 0.7× まで劣化 (vLLM 計測)

ローカル運用は基本バッチ 1 なので**そこは問題ないが、API サーバ的に使うと効果が落ちる**。ターゲット層の明確化 (= バッチ 1 特化) で対処する。

### 3.4 Draft model の調達

EAGLE 系は **target model に対応した draft head の学習が必要**。Medusa も同様:

- ライブラリだけでは完結しない (事前学習済み draft が必要)
- 各モデル (Llama / Qwen / Mistral) ごとに別 draft を用意
- Hugging Face に公開されている draft model に依存

**判断ポイント**: ライブラリの価値は「draft head 学習」と「ランタイム」の両方を含むか、**ランタイムだけに絞るか**。本企画では Phase 1〜3 はランタイム実装に集中、学習側は商用サポートとしてのオプション扱いとする。

### 3.5 SD 特有の実装難所

probe では触らない領域:

- **Tree attention**: draft が複数経路を提案する場合の attention 構造
- **Verification の数学的正しさ**: 確率分布が一致することの保証 (rejection sampling)
- **KV キャッシュの巻き戻し**: draft が rejected された時のキャッシュ復元
- **Multi-head spec** (Medusa): 複数 head の同時予測と統合

これらは実装でハマる可能性が高い。**candle / mistral.rs にこれらの primitive がないので、自前で書く必要がある**。Sprint S1 Day 1 で primitive 整備を済ませる前提。

### 3.6 競合動向への依存性

最大の競合は **llama.cpp**。EAGLE-3 PR (#18039) を完成させると abyo-speculate の価値が半減する。

**対応策**:
- llama.cpp PR の進捗を継続監視 (週 1 確認)
- 完成を察知したら戦略変更 (SAGUARO 等の最新手法に集中、または bench harness としての価値にピボット)

---

## 4. 採用する論文と手法

優先順位を絞る。**全部実装するのは無理**なので、ROI 順に階段状に進める。

### Phase 1: 基本 SD + Medusa (最優先)

- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023) — 古典
- Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" (ICML 2024)

**理由**: 最もシンプル、実装の足場になる。Medusa は draft model 不要 (target model に head を増やすだけ) で、ローカル運用に向く。

**期待スピード**: 1.5〜2× / **実装期間**: 5〜7 日

### Phase 2: EAGLE / EAGLE-2 / EAGLE-3

- Li et al., "EAGLE / EAGLE-2 / EAGLE-3" (2024-2025)

**理由**: Medusa より高速化率が高い (2〜3×)。コミュニティで最も注目。

**注意**: バッチサイズ依存性が大きい。ローカル (バッチ 1) に最適化する余地あり。

**実装期間**: 5〜7 日 (Phase 1 の上に乗せる)

### Phase 3: SAGUARO または Recurrent Drafter

- "Speculative Speculative Decoding (SAGUARO)" (2026) — 30% faster than strongest baseline (**要事前確認**: 論文タイトル / 発表年 / 著者を着手前に裏取り)
- Apple, "Recurrent Drafter" (2025)

**理由**: 2026 年時点での SOTA。**まだどの統合ライブラリも対応していない**。先行者利益が大きい。

**実装期間**: 5〜7 日

### Phase 4 以降: SWIFT / DART / Lookahead Decoding 等

時間が許せば実装。差別化軸の本命にはしない。

---

## 5. 実装計画

### Sprint S1: 基盤 + Phase 1 (Week 6 後半、5〜7 日)

abyo-llm-probe v0.1 が完成した後に着手。

| Day | タスク |
|-----|--------|
| 1 | リポジトリセットアップ、candle 統合、KV キャッシュ操作 primitive 実装 |
| 2 | 基本 Speculative Decoding (draft model 別ロード方式) |
| 3 | Verification の正しい実装 (rejection sampling) |
| 4 | Medusa (multi-head 拡張) |
| 5 | Llama 3.1 8B での動作確認、ベンチマーク |
| 6 | Qwen 2.5 7B、Mistral 7B 対応 |
| 7 | ドキュメント、README、ブログ草稿 |

### Sprint S2: Phase 2 (Week 7、5〜7 日)

EAGLE-2 / EAGLE-3 の実装。**Phase 1 のバッチサイズ 1 最適化を引き継ぐ**。

### Sprint S3: Phase 3 (Week 8 前半、5〜7 日)

SAGUARO または Recurrent Drafter の実装。**SOTA 追従**として価値が高い。

### 合計工数

| シナリオ | 日数 |
|----------|------|
| 楽観 (probe ノウハウフル活用) | 12〜15 日 |
| 標準 | 15〜21 日 |
| 悲観 (tree attention 等で詰まる) | 21〜28 日 |

AI Forecaster と並走させる場合、AI Forecaster は週末枠に。

---

## 6. ライブラリ設計

```rust
use abyo_speculate::{SpeculateEngine, Method};

let engine = SpeculateEngine::builder()
    .target_model("llama-3.1-8b-instruct")
    .method(Method::EAGLE3)
    .draft_path("path/to/eagle3-llama-3.1-8b")
    .build()?;

let output = engine.generate(prompt, /* max_tokens = */ 500)?;
// → 通常の generate と同じ I/F、内部で SD が動く
```

API は `transformers.generate()` 互換にして移行コスト最小化。

### 設定プリセット

```rust
SpeculateEngine::preset_for("llama-3.1-8b")?
// → Llama 3.1 8B 用の最適設定 (手法、draft、acceptance threshold) が自動適用
```

これで「**何も考えずに使えば 2〜3× 速くなる**」体験を作る。

### バックエンド対応

| 優先度 | バックエンド | 備考 |
|--------|-------------|------|
| 第一 | candle | Pure Rust、エコシステム最大 |
| 第二 | mistral.rs | v0.2 以降で検討 |
| 検討 | llama.cpp バインディング | Rust から呼ぶ形、native は llama.cpp 側 |

---

## 7. ベンチマーク戦略

**正直なベンチマーク**を出す。これが信頼の核。

### ベンチ対象

- Llama 3.1 8B Instruct
- Qwen 2.5 7B Instruct
- Mistral 7B v0.3
- Phi-3.5 mini

### 測定項目

- バッチサイズ 1 での tok/s
- 各手法 (baseline vs Medusa vs EAGLE-2 vs EAGLE-3 vs SAGUARO)
- 各タスク (chat / coding / translation / long context)
- 各 GPU (RTX 4070 Ti Super、可能なら借りた A100)

### 出す数字 / 出さない数字

| | OK | NG |
|---|----|----|
| ✅ 範囲明示 | 「Llama 3.1 8B + チャットで 2.4×、コードで 2.1×」 | |
| ✅ モデル別 | 「Qwen 2.5 7B + 長文生成で 1.8×」 | |
| ❌ 全称命題 | | 「全タスクで 3×」 |

**速くならないケースも明記**:

- MoE モデル (速くならない / 遅くなる)
- 高温度サンプリング (rejection 率が上がる)
- 短い出力 (オーバーヘッドが相対的に大きい)

---

## 8. マネタイズ

abyo-llm-probe と同じ構造。直接収益はゼロ前提、**ブランディング投資**として位置づける。

### 短期 (OSS 公開〜半年)

- crates.io 公開、GitHub Star 獲得
- ブログ「Rust で Speculative Decoding を実装、ローカル LLM を 2〜3× 高速化」
- HN / r/LocalLLaMA / r/rust で発信

### 中期 (半年〜1 年)

- 商用サポート契約 (特定モデル向けの draft head 学習サービス含む)
- 企業向けカスタマイズ (特定 GPU、特定モデル最適化)

### 長期 (1 年以上)

- ローカル LLM サービス事業者からの採用
- AWS Bedrock / Azure OpenAI 競合の Rust 推論基盤としての採用

### Ferro 売却バリュエーションへの効果

- abyo-llm-probe + abyo-speculate で「**ローカル LLM 推論基盤の専門家**」ポジション確立
- Ferro 売却交渉時の **技術評価** に直接効く (「LLM 推論基盤も書ける検索エンジン作者」)
- 二次買収候補 (NVIDIA / Cerebras / Groq / Together AI / Anyscale 等) の関心が出る可能性

---

## 9. 競合分析

| 競合 | 言語 | ターゲット | 強み | 弱み |
|------|------|----------|------|------|
| vLLM | Python | データセンター | 機能網羅、コミュニティ大 | バッチ重視、ローカル不向き |
| SGLang | Python | データセンター | 高速、構造化生成 | 同上 |
| TensorRT-LLM | C++ | NVIDIA データセンター | 最速 | NVIDIA 縛り、複雑 |
| llama.cpp | C++ | ローカル | 普及、軽量 | EAGLE-3 PR 進行中 / SD 限定的 |
| candle 内蔵 | Rust | ローカル | Rust ネイティブ | SD 未対応 |
| **abyo-speculate** | **Pure Rust** | **ローカル特化 (バッチ 1)** | **Rust 初の SD 統合実装** | **新規参入、要 draft 調達** |

**最大の競合は llama.cpp**。継続監視必須 (3.6 節参照)。

---

## 10. 想定ユーザ・採用シナリオ

### 採用候補

1. **個人開発者**: ローカル LLM でエージェント / ボット / ツールを作る層
2. **Rust LLM サービス開発者**: Tauri、CLI、組み込み用途
3. **エージェント運用者**: Claude Code 風の自前エージェントを動かしたい開発者
4. **研究者**: SD 各手法の比較実験を Rust で行いたい層
5. **ローカル LLM SaaS**: GPU コスト削減目的

### 既存ユーザ層との接続

- abyo-llm-probe ユーザはそのまま流入 (同じローカル LLM 利用層)
- Godot MCP Pro 顧客の中で「ゲーム NPC をローカル LLM で動かしたい」層
- TalkBuddy のオフラインモード (将来)

---

## 11. 成功基準

| レベル | 基準 |
|--------|------|
| 最低限 (実験成立) | Phase 1 (Medusa) 動作 / ベンチマーク取得 / crates.io 公開 / ブログ + HN 投稿 |
| 部分成功 (プロダクト立ち上がり) | GitHub Star 200+ (半年) / OSS 採用例 10 リポ+ / HN フロントページ入り |
| フル成功 | llama.cpp に勝てる Rust エコシステム定番化 / 商用採用 1 社+ / SaaS からの採用 |

---

## 12. 関連論文インデックス

### 古典

- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (Google, 2023)
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (DeepMind, 2023)

### 主要手法 (実装対象)

- Cai et al., "Medusa" (ICML 2024)
- Li et al., "EAGLE / EAGLE-2 / EAGLE-3" (2024-2025)
- Zhang et al., "Speculative Speculative Decoding (SAGUARO)" (2026) — **要事前確認**
- Apple, "Recurrent Drafter" (2025)

### 関連手法 (時間が許せば)

- Apple, "Speculative Streaming" (Apr 2024)
- "DART (Diffusion-Inspired)" (Jan 2026)
- "SWIFT (Self-Speculative)" (2024)
- "Lookahead Decoding" (2023)

### バッチ・大規模対応

- "Decoding Speculative Decoding" (2024) — ベンチマーク総括
- "Efficient Speculative Decoding for Llama at Scale" (2025) — 大規模対応の課題
- "SwiftSpec" / "SpecBranch" (2025)

### 産業界実装の現状

- vLLM speculative decoding documentation
- TensorRT-LLM Speculative Sampling
- IBM "MLP Speculator" (Wertheimer et al., 2024)

---

## 13. 横断的な接続

### 統合ロードマップとの関係

| Sprint | 本筋 | 並行 |
|--------|------|------|
| Sprint 1 (Week 1-2) | ExaLogLog + abyo-filters | - |
| Sprint 2 (Week 3-4) | FerroSearch 強化 | LLM Hidden State Bench → abyo-llm-probe |
| Sprint 3 (Week 5-6) | FerroStream 差別化 | abyo-llm-probe v0.1 公開、**abyo-speculate Phase 1** |
| Sprint 4 (Week 7-8) | AI Forecaster | **abyo-speculate Phase 2-3** |

### abyo-llm-probe との関係

- **同じローカル LLM 層がターゲット** (probe = encoder 用途、speculate = generator 用途)
- 実装基盤を共有 (candle / mistral.rs のラッパー、ベンチ基盤)
- 「abyo software のローカル LLM ライブラリ群」というブランディング

### Ferro / FerroSearch との関係

直接の依存はないが、**FerroSearch の LLM Query Expansion 機能 (Sprint C 候補) と接続可能**。FerroSearch がローカル LLM でクエリ拡張する際に abyo-speculate で高速化、というシナリオ。

### abyo-crdt との関係

完全に独立。並行進行可能。

---

## 14. 着手前にやること

- [ ] crates.io 名前空間確保: `abyo-speculate`、`abyo_speculate`
- [ ] GitHub リポジトリ作成: `abyo-software/abyo-speculate` (Apache-2.0 / MIT dual)
- [ ] llama.cpp の EAGLE-3 PR (#18039) の最新進捗確認
- [ ] SAGUARO 論文の裏取り (タイトル / 著者 / arXiv ID)
- [ ] 主要モデルの draft head (Hugging Face 上) の調査
  - [ ] Llama 3.1 8B 用 EAGLE / Medusa head の有無
  - [ ] Qwen 2.5 7B 用 EAGLE / Medusa head の有無
  - [ ] Mistral 7B 用 EAGLE / Medusa head の有無
- [ ] candle の現状確認 (KV キャッシュ操作 API、tree attention 対応状況)
- [ ] abyo-llm-probe v0.1 の実装完了 / ノウハウ整理

---

## 15. Sprint S1 着手時の最初のプロンプト (Claude Code 用)

```
@abyo_speculate_plan.md を読んで、abyo-speculate Sprint S1 (Phase 1) を開始してくれ。

タスク:
1. リポジトリ初期化 (abyo-software/abyo-speculate)
2. candle 統合、Llama 3.1 8B Instruct を BF16 でロードできること
3. 基本 Speculative Decoding 実装 (Leviathan 2023 アルゴリズム)
   - draft model: TinyLlama-1.1B または Qwen 2.5 0.5B
   - target model: Llama 3.1 8B
   - rejection sampling の正しい実装
4. ベンチマーク (baseline vs SD の tok/s)

ライセンスは Apache-2.0 / MIT dual。authors は abyo software, LLC。
バックエンドは candle 第一優先、mistral.rs は v0.2 以降で検討。
GPU は RTX 4070 Ti Super 16GB。
```

---

## 16. まとめ

abyo-speculate は abyo-llm-probe に続く **abyo software の LLM ライブラリ第二弾**。

正直に書くと:

- 「Rust で書きました」だけでは差別化にならない (vLLM / SGLang / llama.cpp が既に対応)
- バッチサイズ 1 のローカル運用 + 主要オープンモデル特化、で勝負する
- llama.cpp の EAGLE-3 PR が完成すると価値が半減するので、新手法 (SAGUARO 等) 追従が必須
- MoE モデル等で速くならないケースを正直に明記する

**直接収益はゼロ前提、Ferro 売却バリュエーション向上のブランディング投資**として位置づけ、abyo-llm-probe との二段ロケットで「ローカル LLM 推論基盤の専門家」ポジションを確立するのが目的。
