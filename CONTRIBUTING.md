# 開発と再現

## 通常の開発環境

Python 3.11で仮想環境を作り、`requirements-dev.txt`をインストールしてください。通常の開発・テストにはモデル取得もChromaDBも不要です。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m compileall -q app.py lab_app.py src tests tools
.\.venv\Scripts\python.exe tools/check_publication.py --include-untracked
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tests/test_lab_launchers.ps1
```

最後のコマンドはWindows専用のフェイクによる所有プロセス検査で、実際のサーバーを停止しません。GitHub ActionsはWindows・Ubuntuで通常のテスト、静的解析、公開ファイル検査を実行します。

## テストの境界

- 比較版のUIは`tests/test_lab_app.py`。偽のモデルで説明・出典・切り替えを確認します。
- ローカルモデルの結合テストは明示実行用で、通常はスキップします。
- ChromaDBのない環境では、旧`app.py`の画面8件と旧保存方式1件をスキップします。比較版のテストを除外するものではありません。
- 旧版も再現する場合だけ、[既知のリスク](SECURITY.md)を確認して`requirements-legacy.txt`を使ってください。通常のCIには含めません。
- テスト成功は科学的正確性・実機性能・使いやすさの証明ではありません。

## 変更するとき

1. 文書本文・質問を外部APIへ送らないこと。モデルの自動取得や外部接続の追加は別の設計判断として扱います。
2. 回答方式、検索条件、資料、モデルの変更を混ぜず、比較条件を記録してください。
3. 失敗した回答は、個人情報のない自作データで再現してください。引用一致を正答率と呼ばないでください。
4. ユーザーの資料、索引、FAQ、モデル、評価ログをコミットしないでください。公開前チェッカーの成功後も差分を目視で確認します。
5. AGPLおよび依存物・モデルの利用条件と、データの権利を確認してください。

## 今後の優先課題

1. 質問の対象と出典の対象・数値の結び付きの検証。現行の確認モデルが見逃す事例を含める。
2. 説明の不足・原本にない目的の付加・過剰拒否を、未見データと人手で評価。
3. 同一条件での待ち時間と確認処理の効果の比較。現在の開発測定から高速化倍率を主張しない。
4. 読み手に合う文章の改善。実際の職員・来館者による評価は事前の調整が必要。

新しいモダリティやエージェント機能を増やす前に、上記の品質を測れる状態を保ちます。
