# Third-party notices

このRepositoryは、次の主要な第三者packageを利用します。
表は2026-08-20時点で、各packageのmetadataと公式情報を確認した記録です。
各packageには、それぞれのlicenseが適用されます。

| Package | Version | License | 用途 |
|---|---:|---|---|
| Streamlit | 1.60.0 | Apache-2.0 | Web UI |
| ChromaDB | 1.5.9 | Apache-2.0 | 旧版の保存方式の再現のみ（任意） |
| PyMuPDF | 1.28.0 | GNU AGPL-3.0 or Artifex commercial license | PDF text extraction |
| HTTPX | 0.28.1 | BSD-3-Clause | local Ollama HTTP client |
| PyYAML | 6.0.3 | MIT | configuration loading |
| pytest | 9.0.3 | MIT | development tests |
| Ruff | 0.16.3 | MIT | linting |

PyMuPDFの公式license説明では、GNU AGPLまたはArtifexのcommercial licenseが
適用されます。この依存関係との整合を明確にするため、本Repository自体も
GNU Affero General Public License v3.0で公開します。商用製品へ組み込む場合は、
PyMuPDFを含む各依存関係の条件を利用者自身で確認してください。

完全なtransitive dependency一覧は、対象環境で次を実行して確認できます。

```powershell
python -m pip install pip-licenses
pip-licenses --format=markdown --with-urls
```

この文書はlicense情報の整理であり、法的助言ではありません。

## Museum Evidence Labのモデル（2026-09-21確認）

比較版の通常実行は`requirements-lab.txt`を使用し、ChromaDBを使用しません。
通常の開発環境もChromaDBを含みません。旧版の再現用にだけ`requirements-legacy.txt`を用意しています。[既知のリスク](SECURITY.md)を確認し、比較版の実行経路と区別してください。

モデルの重みはこのGitリポジトリへ同梱せず、ローカルOllamaから読み込みます。
下記は配布元の表示の整理であり、全依存物の法的監査ではありません。

| モデル | 位置づけ | 配布元の条件・確認先 |
|---|---|---|
| `qwen3:4b-instruct-2507-q4_K_M` | 比較版の標準回答モデル | 元モデルはApache-2.0。[Qwen公式モデルカード](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)、[Ollama配布タグ](https://ollama.com/library/qwen3:4b-instruct-2507-q4_K_M) |
| `qwen3.5:2b` | 比較候補 | 元モデルはApache-2.0。[Qwen公式モデルカード](https://huggingface.co/Qwen/Qwen3.5-2B) |
| `embeddinggemma` | 検索用モデル | Gemmaの利用条件を別途確認。[公式モデルカード](https://ai.google.dev/gemma/docs/embeddinggemma/model_card)、[Gemma利用条件](https://ai.google.dev/gemma/terms) |

量子化ファイル、トークナイザー、アプリ本体、Python依存関係にはそれぞれの条件が適用されます。
アプリのAGPL-3.0をモデルの利用条件へ読み替えないでください。施設へモデルを移送・再配布するときは、
その時点で使用する配布物のライセンス本文・必要な通知を確認し、同梱してください。
