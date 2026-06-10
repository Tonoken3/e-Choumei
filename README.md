# SPL: 自給自足勇者

AI勇者が小さな島で1年を生き抜く、ターミナル用の自給自足サバイバルRPGです。

プレイヤーは勇者を直接操作してもいいし、勇者の判断を見守る観戦者になっても構いません。畑を耕し、食料を集め、道具を作り、天候と季節に向き合いながら、少しずつ生活の足場を作っていきます。

この作品は、LLMに世界を自由生成させるゲームではありません。島のルール、在庫、体力、作物の成長、クラフトの成否はすべてPython側のシミュレーションが決めます。LLMや同梱のローカルAIは、勇者の「次に何をするか」だけを考えます。

## 趣旨

SPLは「モデル差し替え型の観戦サバイバル」です。

同じ島、同じseed、同じルールでも、慎重な勇者は冬に備え、無鉄砲な勇者は目先の空腹に追われます。賢さや性格の違いが、そのまま暮らしぶりとして見えるのが狙いです。

LLMを使わなくても遊べるように、標準でローカル勇者AIを同梱しています。LM Studio、vLLM、llama.cpp serverなどのOpenAI互換APIを用意すると、`config/models.toml`のカセット設定でLLM勇者に差し替えられます。

## ゲーム内容

勇者は20x20タイルの島で生活します。

- 1日は12APです。移動、採集、農作業、クラフトなどでAPを消費します。
- HP、満腹度、水分、スタミナ、正気度を管理します。
- 春、夏、秋、冬があり、作物や天候の条件が変わります。
- 食料、木材、石、繊維、魚、粘土、鉱石などを集めます。
- 道具や設備を作ると、生存の選択肢が増えます。
- 眠ると1日が終わり、日記が残ります。

細かい攻略順は伏せます。まずは「今日食べるもの」と「少し先の備え」の両方を見るのが大事です。

## 起動

このリポジトリを取得して、プロジェクトディレクトリで実行します。

```bash
python3 -m spl play
```

通常起動では、同梱のローカル勇者AIが自動で行動します。プレイヤーは観戦しながら、ログ、日記、在庫、マップの変化を眺めます。

## 手動で遊ぶ

勇者を自分で操作したい場合:

```bash
python3 -m spl play --manual
```

よく使うコマンド:

```text
move north
move forest
forage
fish
chop
mine
till
plant turnip
water
harvest
craft stone_axe
build well
cook fish
eat berries
drink
rest
write_diary
sleep
auto
help
quit
```

`auto`を入力すると、そのターンだけローカル勇者AIに任せられます。

## LM Studioで遊ぶ

LM StudioのOpenAI互換サーバーモードを起動し、必要に応じて`config/models.toml`のモデル名を合わせます。

既定では`http://localhost:1234/v1`を見に行きます。

```bash
python3 -m spl play --llm --cassette "Qwen勇者"
```

LLMがJSONを壊したり、知らない行動を返したりしてもゲームは落ちません。そのターンの勇者は混乱し、安全な行動へフォールバックします。

## 高速シミュレーション

1年分を一気に回す場合:

```bash
python3 -m spl simulate --seed 42 --days 112
```

複数seedのローカル勇者を比べる場合:

```bash
python3 -m spl arena --seeds 42,43,44,45 --days 112
```

## ファイル構成

```text
spl/
  core/      世界、勇者、行動裁定、作物、クラフト、イベント
  agent/     観測JSON、ローカル勇者AI、LLM接続、JSONパーサ
  ui/        ターミナルUI
  arena/     簡易リーダーボード
config/      ゲーム設定とLLMカセット設定
data/        作物、レシピ、イベント定義
tests/       回帰テスト
```

## テスト

```bash
python3 -m unittest discover -s tests
```

## 現在の状態

M0からM1相当のプロトタイプです。

- 依存なしのCLI観戦モード
- 手動プレイ
- ローカル勇者AI
- OpenAI互換API接続
- JSON失敗時の混乱フォールバック
- 日記、イベント、クラフト、農業、簡易Arena

TextualによるリッチなTUIや、より演出の濃い日記UIは今後の拡張余地です。
