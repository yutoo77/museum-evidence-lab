# ネットにつながないPCで使うための準備

この比較版は、質問・資料・回答の処理を同じPC内で完結させる設計です。初回のソフトとモデルの入手には、インターネットにつながる準備用PCが必要です。準備したファイルを持ち込めば、実行用PCで毎回ダウンロードする必要はありません。

**現時点で確認したのは、ローカル接続への制限、クラウド無効化、テスト用の通信検証と通常の起動です。ネットワークアダプターを無効にした状態での一連の利用、OS全体の通信監査、科学館の実機への移送は、まだ検証していません。** 「端末から一切通信が出ないことを証明済み」という意味ではありません。正式導入前に、この文書の最後の確認を実機で行ってください。

2026-09-21には、実際のローカルモデルを使う診断でも8項目が成功しました。Pythonプロセス内の外部DNS・接続を拒否した状態で、資料登録、原文回答、資料外質問の拒否、FAQ再利用、資料更新後のFAQ失効を確認しています。記録は`outputs/offline-check/20260921T004238Z-28cdb9df10/report.json`です。この検証でも、別プロセスのOllamaやブラウザ、OSの通信を遮断したわけではありません。

同日の説明方式追加後には、説明生成・確認の2回呼び出しと資料外質問の停止も加えた10項目で成功しました。新しい記録は`outputs/offline-check/20260921T021034Z-0c09003e0b/report.json`です。検証対象は同じPythonプロセス内であり、端末全体の通信遮断や回答の意味の正しさを保証する検査には変えていません。説明品質の制約は[説明方式の確認結果](EXPLANATION_RESULTS.md)を参照してください。

## このPCで普段使うとき

比較版フォルダの `start_lab.cmd` をダブルクリックします。ブラウザで `http://127.0.0.1:8502` が開きます。PowerShellからなら、比較版フォルダに移動して次を実行します。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\start_lab.ps1
```

画面の「AIを準備」を実行すると、モデルを読み込みます。PC起動直後はこの準備に時間がかかります。ブラウザを閉じるだけでは、裏で動くアプリは終了しません。終了するときは `stop_lab.cmd` をダブルクリックするか、次を実行します。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\stop_lab.ps1
```

比較版はWeb画面に8502番、専用Ollamaに11435番を使います。従来版の8501番と11434番は停止しません。番号はPC内の接続窓口で、外部のWebサイトの番号ではありません。

起動スクリプトは、既に動いているプロセスの番号だけでなく、実行ファイル、起動日時、親子関係を確認します。8502番や11435番に別のアプリがいた場合は利用を拒否します。エラーを避けるために、無関係なプロセスを強制終了しないでください。

## 初回だけ行うオンライン準備

用意するものはWindows用Python 3.11・64bit、Windows用Ollama、比較版のソース、Pythonパッケージ、ローカルモデルです。Ollamaは、この比較版で使うモデルを扱え、`OLLAMA_NO_CLOUD=1`によるクラウド無効化を起動ログで確認できる版を使用します。

PythonとOllamaをインストールした後、比較版フォルダで次を実行します。この操作ではパッケージ配布元に接続するため、機密資料を入れる前の準備段階で行ってください。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lab.txt
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\start_lab.ps1 -NoBrowser
```

比較版の実行用依存関係は `requirements-lab.txt` です。従来版用のChromaDBは、比較版の通常実行には不要です。全体の開発テストを行う場合だけ、別途開発用の依存関係が必要になります。

モデルの入手は、準備段階で明示的に行います。以下は専用Ollamaを接続先にして、最小構成の検索モデルと回答モデルを入手する例です。

```powershell
$previousOllamaHost = $env:OLLAMA_HOST
try {
    $env:OLLAMA_HOST = 'http://127.0.0.1:11435'
    ollama pull embeddinggemma
    ollama pull qwen3:4b-instruct-2507-q4_K_M
    ollama list
} finally {
    $env:OLLAMA_HOST = $previousOllamaHost
}
```

追加の比較をする場合のみ、同じ方法で `qwen3.5:2b`、`qwen3:1.7b` を用意します。画面に候補として載っていても、未インストールのモデルをアプリが自動でダウンロードすることはありません。

モデルの実サイズは種類・量子化・配布版によって変わります。複数モデルでは合計数GB以上になるので、`ollama list`と移送フォルダの実サイズを確認し、展開用の空き領域も用意してください。モデルファイルの容量と、実行中に必要なメモリ・GPUメモリは同じではありません。

## 別のオフラインPCへ持ち込むとき

### 1. パッケージ一式を集める

実行先と同じWindows・64bit・Python 3.11の準備用PCで行います。以下は比較版だけを入れた準備用環境を作り、その時点で解決した全依存パッケージを固定して保存する例です。

```powershell
New-Item -ItemType Directory -Path work\offline-bundle\wheelhouse -Force
py -3.11 -m venv work\offline-prep
.\work\offline-prep\Scripts\python.exe -m pip install -r requirements-lab.txt
.\work\offline-prep\Scripts\python.exe -m pip freeze |
    Set-Content -Encoding ascii work\offline-bundle\requirements-lock.txt
.\work\offline-prep\Scripts\python.exe -m pip download --only-binary=:all: `
    --dest work\offline-bundle\wheelhouse `
    -r work\offline-bundle\requirements-lock.txt
```

最後のコマンドが失敗した場合は、必要な配布ファイルがそろっていません。そのまま移送を進めず、同じOS・Python版に対応する配布ファイルを準備してください。完全な再現性を得るには、この固定ファイルと実際の配布ファイルをセットで残します。上位パッケージだけを固定した元のrequirementsファイルでは、後日取得する間接依存の版まで同じになる保証はありません。

### 2. アプリ・モデル・インストーラーをまとめる

移送するアプリには、少なくとも次を含めます。

- `lab_app.py`、`lab_config.yaml`、`requirements-lab.txt`。
- `src`フォルダ全体と、`.streamlit/config.toml`。
- `start_lab.cmd`、`start_lab.ps1`、`stop_lab.cmd`、`stop_lab.ps1`、`lab_process_helpers.ps1`。
- `sample_docs`、`benchmarks/lab_v2`、説明書、`LICENSE`、`NOTICE`、`THIRD_PARTY_NOTICES.md`。
- 前の手順で保存した`requirements-lock.txt`と`wheelhouse`。

`.git`、`.venv`、`.streamlit/secrets.toml`、`.env`、キャッシュ、一時出力、不要な`data`はコピーしません。この開発フォルダはGit worktreeの場合があり、`.git`をコピーすると元PCの場所を参照してしまいます。仮想環境もPCごとに作り直します。

モデルはOllamaのモデル保存先から、`blobs`と`manifests`を含む必要なファイルをまとめます。標準のWindows保存先はユーザーフォルダ内の`.ollama\models`ですが、`OLLAMA_MODELS`を設定している場合はその場所です。モデルの更新やダウンロードが動いていない状態でコピーしてください。モデル置き場全体を移す場合は、不要なモデルまで含まれていないか確認します。

Python 3.11・64bitと、準備用PCと同じOllama版のインストーラーも必要です。移送物のハッシュ値を保存し、到着後に一致を確認すると、コピー不良の検出に役立ちます。

```powershell
Get-ChildItem .\work\offline-bundle -Recurse -File |
    Get-FileHash -Algorithm SHA256 |
    Format-Table Path, Hash -AutoSize
```

USBメモリなどの持ち込み・ウイルス検査は施設の運用に従ってください。実資料を移す場合は、資料の持ち込み権限も別途必要です。アプリ本体、Pythonパッケージ、各モデルではライセンスが異なります。`LICENSE`などを同梱し、モデルの配布元に示された利用条件と再配布条件を、使用する版ごとに確認してください。モデルを手元で使えることと、そのまま第三者へ再配布できることは同じではありません。

### 3. オフラインPCでセットアップする

持ち込んだインストーラーからPythonとOllamaをインストールします。アプリを、OneDriveなどの自動同期先に入っていないローカルフォルダへ置きます。モデルファイルは、このPCのOllamaが参照するモデル保存先へ配置します。空き容量とファイルコピーの完了を確認してください。

アプリフォルダへ移動し、パッケージを移送ファイルだけからインストールします。以下の`offline-bundle`は持ち込んだ場所に合わせてください。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-index `
    --find-links .\offline-bundle\wheelhouse `
    -r .\offline-bundle\requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\start_lab.ps1
```

`--no-index`は、インターネットのパッケージ一覧を使わない指定です。不足がある場合は、その場でネット接続して補うのではなく、準備用PCで移送物を作り直してください。

## アプリが守る範囲と、端末側で守る範囲

アプリのモデル通信は、設定で許可したPC内のOllamaだけです。外部IP・通常のホスト名・URL転送を拒否し、環境変数のHTTPプロキシを使いません。モデルも、許可名、ローカルの識別値、ローカルモデルのメタデータを確認してから質問や資料を送ります。起動スクリプトは、専用Ollamaにクラウド無効化を設定し、起動ログの確認ができない場合は利用を拒否します。

画面は質問、回答、資料本文を通常の文字として表示します。回答に外部画像のMarkdownが含まれても、その画像を読み込みません。Streamlitの利用統計送信は無効です。通常の利用ログは初期設定で無効で、有効にしても質問本文・回答本文・思考内容を保存せず、処理時間や状態だけを記録します。

ただし、登録資料は検索のためにローカルの`data/lab`へ保存します。暗号化された保管庫ではありません。プロセスの起動ログも`data/lab/processes`に保存されます。比較評価を実行した場合は、`outputs/lab-comparisons`に質問・回答を含む評価成果物が残ります。これらはGitへ追加しない設定ですが、Windowsのバックアップやクラウド同期、他のソフトによる読み取りまでは防げません。

また、ブラウザ拡張、OSやOllamaデスクトップアプリの更新確認、ウイルス対策ソフト、同じPC内の別プロセスによる通信は、このアプリの設定だけでは止まりません。機密性の要件が「外向きの通信を許可しない」であれば、施設側でネットワークの物理切断や外向き通信の遮断を実施します。ループバック通信まで遮断するとアプリ自身が動かなくなるため、PC内の通信は残します。

このアプリには利用者認証を実装していません。127.0.0.1で待ち受けるのは、そのPC内で使うためです。LAN公開、ポート転送、共有サーバー化はこの運用の対象外です。ローカル管理者や侵害済みOSからデータを守る仕組みでもありません。実行中のモデルの入れ替えも避けてください。

## 本番導入前のオフライン確認

施設の管理者と相談して実施します。以下は未実施の受け入れ確認項目であり、達成済みの保証ではありません。

1. 実資料の代わりに公開可能なサンプルで準備する。
2. 許可された方法で外向き通信を遮断し、PCを再起動する。
3. `start_lab.cmd`から起動し、モデルの準備を完了する。
4. 資料の登録、初めての質問、根拠の表示、資料外質問への回答抑制を確認する。
5. コピーや再起動後も同じことができるか確認する。
6. 管理者の通信監査で、アプリ・Ollama・ブラウザからの不要な外向き接続を確認する。
7. `stop_lab.cmd`で比較版だけが終了し、他のアプリに影響しないことを確認する。

オフライン化は回答の正しさを保証しません。資料の承認・更新日・数値の確認を続け、回答内容が原文に合うかを職員が確認できる運用と組み合わせてください。

## 補助的なローカル通信診断

専用Ollamaと必要なモデルを準備した状態で、次のCLIも利用できます。通常のアプリや速度比較と同時に質問処理を行わず、検査だけを実行してください。

```powershell
.\.venv\Scripts\python.exe tools\lab_offline_check.py
```

このCLIは、そのPython処理内で外部DNS・外部接続と、11435番以外への接続を拒否します。その状態で、同梱の架空資料5件の登録、代表質問への原文回答と説明案、資料外質問の拒否、架空FAQの承認・再利用・資料更新後の失効を確認します。説明案は生成・確認の2回呼び出しと引用の存在を検査しますが、意味の正確性までは評価しません。サンプルは内容の識別値が一致するものに限定され、任意の施設資料を読み込みません。

検査用の索引とFAQは、毎回新しい`outputs/offline-check/日時-識別子`の中に作成されます。通常の`data/lab`は使いません。`report.json`は状態、接続先、処理時間だけを残し、質問文や回答文は出力しません。索引には検査に使った架空資料と架空FAQが残ります。

これはPythonのソケット呼び出しに対する補助検査です。別プロセスのOllama、ブラウザ、OS、ネイティブ拡張や子プロセス全体の通信を監視するものではなく、物理的なオフライン試験の代わりにはなりません。
