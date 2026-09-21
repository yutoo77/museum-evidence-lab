# 公開デモ用 RAGベンチマーク v1

このディレクトリは、架空の科学館資料だけを使ってベンチマーク基盤の読み込み・実行・集計を確認するための、公開可能なスモークデータセットです。

## 重要な位置づけ

`cases.jsonl`の9問はすべて`split: smoke`です。研究評価用の`test`分割ではありません。既存の`evaluation_questions.json`をスキーマ付き形式へ移し、正解根拠を付けた動作確認用データです。

この9問は既存システムの距離しきい値調整やデモに利用済みであるため、未知データに対する性能、汎化性能、研究上の有効性を示す結果として報告してはいけません。

## ファイル

- `case.schema.json`: 1問分のJSON Schema Draft 2020-12
- `cases.jsonl`: 3分類各3問、計9問のスモークデータセット
- `baseline_manifest.yaml`: 既存RAG、モデル、設定、資料集合の再現用記録
- `environment.snapshot`: 初回実測時の直接依存関係を凍結した記録。インストールには使用しない
- `initial_smoke_result.json`: 2026年8月11日の初回ローカル実測要約
- `README.md`: 利用範囲とラベル規則

## 対象資料

正解根拠には、`src.demo_data.DEMO_DOCUMENT_FILENAMES`で登録される次の架空資料だけを使用しています。

- `moon_phase.txt`
- `solar_system.txt`
- `black_hole.txt`
- `planetarium_guide.txt`
- `museum_faq.txt`

これらは`sample_docs/`に同梱されたデモ資料であり、実在する科学館の公式情報ではありません。

## 分類規則

| 分類 | 期待状態 | 回答可能性 | 正解根拠 |
|---|---|---|---|
| `answerable` | `answered` | `YES` | 1件以上 |
| `unrelated` | `low_relevance` | `NOT_RUN` | 空 |
| `insufficient` | `unanswerable` | `NO` | 話題上関連する資料を1件以上 |

`insufficient`の正解根拠は「その資料に答えがある」という意味ではありません。「正しい話題の資料を検索した上で、資料だけでは答えられないと停止する」ことを評価するための関連資料です。

## 読み込み

```python
from src.benchmark import load_benchmark_cases

cases = load_benchmark_cases("benchmarks/rag_v1/cases.jsonl")
```

読み込み処理はスキーマ版、分類と期待値の整合、正解根拠の有無、空文字、重複IDを検査します。JSON Schemaファイル自体を検証する追加パッケージは必要ありません。

## 隔離されたスモーク実行

Ollamaを起動し、プロジェクトのルートディレクトリで次を実行します。

```powershell
python -m src.benchmark.cli --run-id smoke-local-01
```

`outputs/benchmarks/smoke-local-01/`へ専用SQLite、ChromaDB、利用ログ、質問ごとの結果、CSV、集計、実行条件を保存します。通常アプリの`data/`は使用しません。`outputs/`はGit管理外です。

初回実測は9問すべて期待状態と一致しましたが、p95は68.3506秒でした。これは研究精度ではなく、ベンチマーク経路が実機で動くことと、応答時間の改善が次の課題であることを示すスモーク結果です。

## 研究用データセットを作る場合

このファイルの分割名だけを`test`へ変更して流用しないでください。別版として、未調整の質問、固定した資料集合のハッシュ、`dev`とロックした`test`、ラベル作成手順を用意してください。しきい値やプロンプトの調整は`dev`だけで行います。
