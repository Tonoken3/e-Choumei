# SPL: 自給自足勇者

Macのターミナルで遊べる、AI勇者の自給自足サバイバルRPGです。

まずは依存なしで動くローカル勇者AIを同梱しています。LM Studio、vLLM、llama.cpp serverなどのOpenAI互換APIを用意すると、`config/models.toml`のカセット設定でLLM勇者に差し替えられます。

## 起動

```bash
python3 -m spl play
```

手動で遊ぶ場合:

```bash
python3 -m spl play --manual
```

1年分を高速に自動実行する場合:

```bash
python3 -m spl simulate --seed 42 --days 112
```

## コマンド例

手動プレイでは、次のようなコマンドを入力できます。

```text
move north
move forest
forage
till
plant turnip
water
harvest
craft stone_axe
eat berries
drink
sleep
auto
help
quit
```

## LLMカセット

`config/models.toml`の`base_url`、`model`、`api_key`をOpenAI互換APIに合わせて設定し、次のように起動します。

```bash
python3 -m spl play --cassette "Qwen勇者" --llm
```

LLMがJSONに失敗してもゲームは落ちません。勇者は混乱し、安全行動にフォールバックします。

## テスト

```bash
python3 -m unittest discover -s tests
```
