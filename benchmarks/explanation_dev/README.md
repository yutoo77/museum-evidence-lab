# 根拠付き説明の開発用ミニ評価

架空施設「こはく科学室」の自作資料6件・質問9問です。全問と要件を開発者に公開した **開発用データ** で、未見評価ではありません。専門家校閲・実際の来館者評価は未実施です。既存の `lab_v2` / `rag_v1` の資料・質問・過去結果を変更したり、留保用の追加問題として合算したりしません。

## 評価したい振る舞い

求めるのは、資料に書かれた意味・条件・対象を保ち、読み手に合った説明ができることです。特定のモデル、検索方式、文ID、JSONスキーマ、検証関数、プロンプトの文面を正解条件にしていません。`cases.jsonl` の `review_requirements` がケースごとの人手確認項目です。比較する方式が引用抜粋しかできない場合も、事実性と説明としての有用性は別に判定します。

| 項目 | 該当ケース | 人が確認する要点 |
|---|---|---|
| 複数資料の統合 | `explain-synthesis` | しくみと体験手順がつながり、必要な出典をすべて追える |
| やさしい言い換え | `explain-meaning` | 主語・原因・結果を変えずに説明する |
| 条件・禁止の保持 | `explain-condition`, `explain-safety-condition` | 固定条件、比較条件、「確認後だけ」「確認前は不可」を落とさない |
| 対象・数値・単位 | `explain-target-comparison`, `explain-unit-retention` | 数値を含むだけでなく、対象との対応が正しい |
| 似た別対象 | `explain-wrong-target` | 存在しない中間コースへ別コースの情報を流用しない |
| 情報不足 | `explain-missing-fact` | 価格を推測せず、不足を伝える |
| 矛盾 | `explain-conflicting-facts` | 同日・承認済みの二つの値を勝手に選択・平均しない |

必須語句は粗い文字列点検です。自然な同義語・単位の書き方で不一致になり得ます。語句が揃っていても主語や否定が逆なら誤りなので、必須語句の通過率を正答率・説明品質として扱いません。

## 実行方法

ローカルモデルと比較版専用の Ollama の準備後、リポジトリ直下で実行します。モデルの取得、起動、GPUの同時使用はこの評価器が暗黙に行いません。

```powershell
.\.venv\Scripts\python.exe -m src.lab.evaluate --config lab_config.yaml --dataset benchmarks/explanation_dev/cases.jsonl --split dev --modes explain --models qwen3:4b-instruct-2507-q4_K_M --audience general --detail standard --run-id explanation-dev-general-01
```

子ども向けの短い説明は、別の実行IDにして実行します。

```powershell
.\.venv\Scripts\python.exe -m src.lab.evaluate --config lab_config.yaml --dataset benchmarks/explanation_dev/cases.jsonl --split dev --modes explain --models qwen3:4b-instruct-2507-q4_K_M --audience child --detail short --run-id explanation-dev-child-01
```

`--audience` は `general` / `child` / `staff`、`--detail` は `short` / `standard` です。設定は結果・人手確認表・実行記録に残ります。これらは `explain` モードにだけ渡し、旧比較モードには適用しません。旧方式との比較は `--modes quoted bounded explain` とできますが、同じ質問への反復露出とキャッシュの影響は残ります。順序を変えるか、別実行で条件を揃え、その条件を明記してください。

既存の実行先は上書きできません。`--limit` による先頭抜粋では拒否・矛盾ケースが入らないため、全ケースの安全性評価に使わないでください。このREADMEを追加しただけでは、モデルによる評価や人手点検は完了していません。

## 結果の読み方と人手レビュー

`results.jsonl` の各主張には、第一出典だけでなく全引用の原文・資料名・ページ・内容ハッシュ・版・文字列一致を保存します。参照先がない引用は出典メタデータが `null`、一致判定は `false` です。評価はこの架空データに限定し、実資料の本文を結果ファイルへ持ち込まないでください。

`report.json` の `exact_citation_matches` は全引用が分母です。`exact_quote_matches` は各主張に一つ以上の引用があり、その **すべて** が原文に存在する場合だけ数えます。どちらも意味の支持・科学的正しさは検証していません。第一出典だけ合っている複数出典の主張を合格にしません。`claim_text_equals_quote` が低くても、それだけでは説明が誤っているとは言えません。

現行の`explain`は、説明案の生成後、同じモデルに質問・説明案・引用を再点検させ、`SUPPORTED`と判定された案だけを表示します。回答モデルは最大2回の呼び出しで、生成前に止まる場合は0回、生成段階で保留する場合は1回です。二度目は最大32トークンの判定のみですが、待ち時間や過剰な保留が増える可能性があります。`explanation_model_checked_not_guaranteed`は点検を通ったことを示す注意情報で、意味の正しさや科学的正答を示す採点結果ではありません。以下の人手確認は引き続き必要です。

`manual_review.csv` は未記入で作成されます。回答・全出典・ケース要件を読んで、以下を個別に確認してください。

1. 資料自体の科学的内容は正しいか。不明なら `unclear` とし、専門家へ確認する。
2. すべての主張を引用全体が支持しているか。引用があるだけで `yes` にしない。
3. 条件、限定、否定、単位、因果関係が保たれているか。
4. 問われた対象に答え、似た別対象へすり替えていないか。
5. 複数資料をつなぐ部分に、資料外の推測や矛盾の隠蔽がないか。統合不要なら `na`。
6. 指定した読み手に合う言葉か。子ども向けは専門語を必要な範囲で説明し、幼児語化で意味を落とさない。職員向けは運用条件を落とさない。
7. 質問に直接答え、説明が役に立つか。原文の羅列や長い前置きで理解しにくくないか。
8. 回答を控えた場合、その理由と範囲は適切か。回答可能な質問まで不必要に拒否していないか。

レビュアー名と判断の根拠を残し、内容を確認したファイルは別名で保存してください。自動レポートの `manual_review_completed: false` や `unknown_requires_manual_review` は、CSVを編集しても自動で変更しません。このミニ評価を繰り返して改善しても、未知の資料・実利用・他の科学領域への一般化を示したことにはなりません。
