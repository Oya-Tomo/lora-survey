#import "format.typ": template

#import "@preview/cetz:0.3.2"

#show: template.with(
  main: "LoRA系Fine-tuning手法の調査",
  sub: "Survey of LoRA-variants for Generative Models",
  student: "小山田 智典 (Tomonori Oyamada)",
  teacher: "村上 力 (Riki Murakami)",
)

= はじめに

#par()[
　近年の研究では，事前学習済み大規模モデルが様々なタスクへ応用されている．
HuggingFaceをはじめとする様々なプラットフォームで，事前学習済みモデルの重みが公開されており，
様々なタスクに対してFine-tuning (FT) を行うことで，容易に性能の高いモデルを構築することが可能である．

　一方で，FTにかかる計算リソースがしばしば問題となる．
FTの最も初歩的な手法はFull-Fine-Tuning (FFT) だが，
大規模な事前学習モデルの全パラメタを更新対象とするため多くの計算リソースを必要とする．
この問題を解決するため，近年の手法では学習対象のパラメタを限定・量子化することで計算リソースを削減し，
効率的にFTを行う手法 (Parameter-Efficient Fine-Tuning，PEFT) が提案されている．

　PEFTの中でも，学習パラメタの削減手法は様々なものが提案されているが，
@Mao_2024 では主に以下の2つに分類している．
一つはExtra-Parameter Method (EXP) であり，
Adapter @houlsby2019parameterefficienttransferlearningnlp などのように元のパラメタを完全に固定し，
追加学習用のパラメタを挿入する手法である．
もう一つはIntra-Parameter Method (INP) であり，LoRA @hu2021loralowrankadaptationlarge のように，
元のモデルのパラメタを固定し，そのモデルのパラメタ更新を代替の低ランクな行列で肩代わりする手法である．

　本稿では，PEFTの中でも近年のFT手法のスタンダードとなりつつあるLoRAについての調査を行い，
来年度の研究に向けた基礎知識を整理することを目的とする．
]

#figure(
  align(center)[
    #cetz.canvas(
      {
        import cetz.draw: *
        let highlight(body, color) = {
          place(strike(stroke: 1.1em + color, body))
          body
        }
        content(
          (2.5, 2.3),
          [
            #text(size: 8pt)[
              新しいパラメタを追加するか
            ]
          ],
          anchor: "center"
        )
        line(
          (2.6, 2.1),
          (2.6, 1.6),
          mark: (end: "stealth", fill: black)
        )
        content(
          (2.6, 1.3),
          [
            #text(size: 6pt)[
              しない
            ]
          ],
          anchor: "center"
        )
        content(
          (2.6, 0.5),
          [
            #text(size: 6pt)[
              する
            ]
          ],
          anchor: "center"
        )
        cetz.tree.tree(
          (
            [FT],
            (
              [FFT],
            ),
            (
              [PEFT],
              (
                [EXP],
                (
                  [Adapter],
                )
              ),
              (
                [INP],
                (
                  [LoRA],
                )
              ),
            ),
          ),
          draw-node: (node, ..) => {
            if node.content == [LoRA] {
              content((), text([#node.content], weight: "bold"))
            } else {
              content((), text([#node.content]))
            }
          },
          draw-edge: (from, to, ..) => {
            let (a, b) = (from, to)
            line((a, 0.5, b), (b, 0.7, a))
          },
          direction: "right",
          grow: 1.8,
          spread: 0.6,
        )
      },
    )
  ],
  caption: "Fine-tuning系統分岐",
)

= LoRA

#par()[
　LoRA @hu2021loralowrankadaptationlarge は，@Mao_2024 で分類されたINPの一つである．
この手法は，元のモデルのパラメタを固定し代わりに用意したテンソルで元のモデルの更新を近似することで，
元のモデルの性能を維持しつつ，少ないパラメタの更新のみでFTを行うことが可能である．
]


== LoRAの概要

#par()[
　深層学習における，重み $W in RR^(m times n)$ の更新式を以下のように表す．
( $Delta W$ は重みの更新量)
]

$
W' <- W + Delta W
$

#par()[
　LoRAでは， $Delta W$ を低ランクの行列 $A, B$ の積で近似する．
( $A in RR^(r times n)$，$B in RR^(m times r)$，$r << min(m, n)$ )
]

$
W' <- W + B A
$

#par()[
　学習プロセスでは，元のモデルの重み $W$ を固定し $A, B$ のみ更新ことで，使用する計算リソースを大幅に削減することが可能である．
ここで用いられている $r$ はユーザーが調整可能なハイパーパラメタであり， $A, B$ の制限ランク値である．
パラメタ数削減のため， $r$ は通常 $min(m, n)$ よりも十分小さく設定される．
また，タスクの難易度によっても適切な $r$ の値は異なるため，調整が必要である．
]


#figure(
  align(center)[
    #set text(size: 8pt)
    #cetz.canvas({
      import cetz.draw: *

      content(
        (-2.3, 4),
        [#text(size: 12pt, font: "IPAGothic")[h]],
        anchor: "south"
      )
      rect(
        (-2, 4), (2, 4.4), fill: yellow
      )
      rect(
        (-3, 1), (-0.2, 3),
        fill: blue
      )
      content(
        (-1.5, 2),
        [
          #text(size: 12pt, fill: white)[
            $W in RR^(m times n)$
          ]
        ],
        anchor: "center"
      )
      line(
        ..(
          (1, 2.1),
          (2, 2.1),
          (3, 3),
          (0.2, 3)
        ),
        close: true,
        fill: orange,
      )
      content(
        (1.6, 2.6),
        [
          #text(size: 10pt, fill: white)[
            $B in RR^(m times r)$
          ]
        ]
      )
      line(
        ..(
          (0.2, 1),
          (3, 1),
          (2, 2),
          (1, 2),
        ),
        close: true,
        fill: orange,
      )
      content(
        (1.6, 1.4),
        [
          #text(size: 10pt, fill: white)[
            $A in RR^(r times n)$
          ]
        ]
      )
      content(
        (0, 3.7),
        [
          #text(size: 12pt, fill: black)[
            $+$
          ]
        ]
      )
      line(
        (1, 0.5),
        (1.5, 0.9),
        mark: (end: "stealth", fill: black)
      )
      line(
        (-1, 0.5),
        (-1.5, 0.9),
        mark: (end: "stealth", fill: black)
      )
      line(
        (1.5, 3.1),
        (0.5, 3.9),
        mark: (end: "stealth", fill: black)
      )
      line(
        (-1.5, 3.1),
        (-0.5, 3.9),
        mark: (end: "stealth", fill: black)
      )
      rect(
        (-2, 0), (2, 0.4), fill: yellow
      )
      content(
        (-2.3, 0),
        [#text(size: 12pt, font: "IPAGothic")[x]],
        anchor: "south"
      )

      rect(
        (-3.5, -1), (-3, -0.5),
        fill: blue
      )
      content(
        (-2.9, -0.75),
        [
          #text(size: 10pt, fill: black, font: "IPAGothic")[
            Frozen Weight
          ]
        ],
        anchor: "west"
      )

      rect(
        (0, -1), (0.5, -0.5),
        fill: orange
      )
      content(
        (0.6, -0.75),
        [
          #text(size: 10pt, fill: black, font: "IPAGothic")[
            Trainable Weight
          ]
        ],
        anchor: "west"
      )
    })
  ],
  caption: "LoRAのパラメタ構造",
)

== LoRAの利点

#par()[
　LoRAを適用することのメリットとしては，主に以下の3点が挙げられる．

　まず1点目に挙げられるのは，行列 $A, B$ のランク制限による更新パラメタ数の削減である．
ユーザーが $r$ を調整することによって，学習に利用できるGPU等の計算リソースに合わせて規模を変えることができる．
$r$ の調整によっては，効率的にパラメタ数を削減しつつ性能を維持することが可能である．
@hu2021loralowrankadaptationlarge によると，GPT-3 175Bの学習時のVRAM使用量を1.2TBから350GBに削減できたと報告されている．
( $r = 4$，Attentionレイヤのクエリとバリューの投影行列のみにLoRAを適用)


　そして2点目に挙げられるのは，計算コストの低下に伴う学習速度の向上である．
FFTでは更新対象となるパラメタ数が膨大になり学習時間が長くなるため，エポック数不足による精度の低下を招くことがある．
しかし，LoRAを適用することで学習時間を短縮し更に多くのエポックを回すことが可能となり，スピーディーなモデルの開発が可能となる．
@hu2021loralowrankadaptationlarge によると，GPT-3 175BのLoRAを用いた学習速度は，FFTに比べて約25%向上したことが示されている．
(FFTでのスループットは32.5 tokens/s，LoRAでのスループットは43.1 tokens/s)

　3点目に挙げられるのは，ファイルサイズの削減である．
例えば，適応させたい下流タスクが複数ある場合，
FFTを用いると下流タスクごとに大規模な全パラメタを保存する必要があるため，ファイルサイズが膨大になる．
一方でLoRAを用いることにより，下流タスクごとにLoRAで学習したパラメタを保存し，
元のモデルの重みは複数の下流タスク間で共有することが可能となる．
@hu2021loralowrankadaptationlarge によると，GPT3 175Bの学習時チェックポイントのファイルサイズは，
FFTの350GBに対してLoRAでは35MBとなり，約10,000倍縮小されたことが報告されている．
]

/*
#figure(
  cetz.canvas({
    import cetz.draw: *

    rect(
      (0, 1), (1, 2),
      fill: blue
    )
    rect(
      (2, 3), (3, 3.5),
      fill: orange
    )
    rect(
      (2, 2), (3, 2.5),
      fill: orange
    )
    rect(
      (2, 1), (3, 1.5),
      fill: orange
    )
    rect(
      (2, 0), (3, 0.5),
      fill: orange
    )

    content(
      (0.5, 1.5),
      [#text(size: 12pt, fill: white)[
        $W$
      ]]
    )
    content(
      (2.5, 3.2),
      [#text(size: 10pt, fill: white)[
        $A, B$
      ]]
    )
    content(
      (2.5, 2.2),
      [#text(size: 10pt, fill: white)[
        $A, B$
      ]]
    )
    content(
      (2.5, 1.2),
      [#text(size: 10pt, fill: white)[
        $A, B$
      ]]
    )
    content(
      (2.5, 0.2),
      [#text(size: 10pt, fill: white)[
        $A, B$
      ]]
    )
    content(
      (1.3, 3),
      [#text(size: 12pt, fill: black)[
        $+$
      ]]
    )

    line(
      (1, 1.5),
      (2, 3.3),
      mark: (end: "stealth", fill: black)
    )
    line(
      (1, 1.5),
      (2, 2.3),
      mark: (end: "stealth", fill: black)
    )
    line(
      (1, 1.5),
      (2, 1.2),
      mark: (end: "stealth", fill: black)
    )
    line(
      (1, 1.5),
      (2, 0.2),
      mark: (end: "stealth", fill: black)
    )

    line(
      (3, 3.3),
      (4, 3.3),
      mark: (end: "stealth", fill: black)
    )
    line(
      (3, 2.3),
      (4, 2.3),
      mark: (end: "stealth", fill: black)
    )
    line(
      (3, 1.2),
      (4, 1.2),
      mark: (end: "stealth", fill: black)
    )
    line(
      (3, 0.2),
      (4, 0.2),
      mark: (end: "stealth", fill: black)
    )

    content(
      (4.5, 3.3),
      [#text(size: 12pt)[
        Text Generation
      ]],
      anchor: "west"
    )
    content(
      (4.5, 2.3),
      [#text(size: 12pt)[
        Text Classification
      ]],
      anchor: "west"
    )
    content(
      (4.5, 1.2),
      [#text(size: 12pt)[
        Keyword Extraction
      ]],
      anchor: "west"
    )
    content(
      (4.5, 0.2),
      [#text(size: 12pt)[
        More Tasks ...
      ]],
      anchor: "west"
    )
  }),
  caption: "パラメタ共有と下流タスクへの適用",
)

*/

== Adapterとの比較

#par()[
　Adapter @houlsby2019parameterefficienttransferlearningnlp は，
@Mao_2024 で分類されたEXPの一つである．
ここでは，LoRAとAdapterの違いや特徴について比較する．

　まず，Adapterとは，元のモデルのパラメタを固定し，追加学習用のAdapterモジュールをモデルのレイヤ間に挿入する手法である．
挿入するモジュールの構造を Figure 4 に示す．
モジュールは，次元削減レイヤ，非線形レイヤ (ReLU等) ，次元復元レイヤという3つのレイヤからなり，
スキップ接続により残差ネットワークを構成している．
Adapterモジュールはモデルのレイヤ間に挿入されるため，恒等変換性が高く元のモデルの性能を維持しつつ，
下流タスクを残差学習することが可能である．
]

#figure(
  align(center)[
    #cetz.canvas({
      import cetz.draw: *

      line(
        (0, 0),
        (0, 5.5),
        mark: (end: "stealth", fill: black)
      )
      line(
        (0, 0.5),
        (2.3, 0.5),
        (2.3, 4.5),
        (0.35, 4.5),
        mark: (end: "stealth", fill: black)
      )
      circle(
        (0, 4.5),
        radius: 0.3,
        fill: white,
        stroke: black,
        anchor: "center"
      )
      content(
        (0, 4.5),
        [#text(size: 8pt)[
          $+$
        ]],
        anchor: "center"
      )

      rect(
        (-1, 2.2), (1, 2.8),
        fill: gray
      )

      content(
        (0, 2.5),
        [#text(size: 9pt, fill: white, font: "IPAGothic")[
          Nonlinearly
        ]],
        anchor: "center"
      )

      line(
        (-2, 1),
        (2, 1),
        (1, 2),
        (-1, 2),
        close: true,
        fill: orange
      )
      content(
        (0, 1.7),
        [#text(size: 10pt, fill: white, font: "IPAGothic")[
          Feedforward
        ]],
        anchor: "center"
      )
      content(
        (0, 1.3),
        [#text(size: 10pt, fill: white, font: "IPAGothic")[
          down-project
        ]],
        anchor: "center"
      )

      line(
        (-2, 4),
        (2, 4),
        (1, 3),
        (-1, 3),
        close: true,
        fill: orange
      )
      content(
        (0, 3.7),
        [#text(size: 10pt, fill: white, font: "IPAGothic")[
          Feedforward
        ]],
        anchor: "center"
      )
      content(
        (0, 3.3),
        [#text(size: 10pt, fill: white, font: "IPAGothic")[
          up-project
        ]],
        anchor: "center"
      )
    })
  ],
  caption: "Adapterモジュールの構造",
)

#par()[
　AdapterとLoRAの大きな違いは，推論コストにあると考えられる．
Adapterは，モデルのレイヤ間にモジュールを挿入するため，推論時には追加のコストが発生する．
しかし，LoRAで学習したパラメタは $Delta W$ を近似しているため，
パラメタのロード時に元のパラメタに加算することでパラメタ数を増やすことなく推論を行うことが可能である．

　一方で，Adapterはモジュール内部に非線形レイヤを持つため，モデルの表現力を高めることができるという利点がある．
モデルの表現力に焦点を当てる場合，Adapterの方が適していると考えられる．
LoRAとAdapter両方にそれぞれの利点があるため，適用するタスクに応じて選択することが重要である．
]



= LoRA系手法の調査

#par()[
　LoRAは，現在提案されているPEFT手法の中でも，推論時のコストが低いという利点がある．
しかし，近年ではLoRAをベースとした様々な手法が提案されており，
さらなる推論コストの削減や学習効率の向上を目指している．
本稿では，以下のLoRA系手法について調査を行った．
それぞれの手法の概要については以下に述べる．
]

- LoRA @hu2021loralowrankadaptationlarge
- LoRA-FA @zhang2023lorafamemoryefficientlowrankadaptation
- AdaLoRA @zhang2023adaloraadaptivebudgetallocation
- VeRA @kopiczko2024veravectorbasedrandommatrix
- QLoRA @dettmers2023qloraefficientfinetuningquantized

== LoRA-FA

#par()[
　LoRA-FA @zhang2023lorafamemoryefficientlowrankadaptation は，
LoRAの学習パラメタ $A$ を標準正規分布 $cal(N)(0, 1)$ からサンプリングしたランダム値で初期化し固定する．
また，学習時には $overline(B)$ のみを更新することで学習にかかるメモリ使用量を削減する手法である．
( $A_"frozen"$ は固定された行列 $A$ )

$
W' <- W + B A_"frozen"
$

$A$ を $cal(N)(0, 1)$ で初期化する理由として，入力を比較的偏り無く圧縮できるためだとしている．
]

== AdaLoRA

#par()[
　AdaLoRA @zhang2023adaloraadaptivebudgetallocation は，
LoRAの $r$ 値を，レイヤの重要度に応じて動的に割り当てる手法である．

$
W' <- W + P Lambda Q
$

@zhang2023adaloraadaptivebudgetallocation では $Delta W$ を $P, Lambda, Q$ に特異値分解し，
$Lambda$ 内の特異値と，それに対応する $P$ の列，$Q$ の行をtriplet $(P_lambda, Lambda_lambda, Q_lambda)$ として扱う．
モデル内の全てtripletごとに重要度を計算し，重要度が高いものはそのまま残し，低いtripletは0でプルーニングする．
この学習を通して，残すtripletの個数はスケージュルされ $b(t)$ で表される．( $t$ はエポック数)

　この手法により，モデルのレイヤごとに適切な $r$ 値を割り当てることが可能となり，
モデルの性能を保ちつつ計算リソースを削減することが可能となった．
]

== VeRA

#par()[
　VeRA @kopiczko2024veravectorbasedrandommatrix は，
$A, B$ 行列を $cal(N)(0, 1)$ で初期化し固定する．
その代わりに，$Lambda_A, Lambda_B$ を $A, B$ にかけることで代替する．
( $Lambda_A in RR^(r times r), Lambda_B in RR^(m times m)$ )
]

$
W' <- W + Lambda_B B Lambda_A A
$

#par()[
ここで，$Lambda_A, Lambda_B$ の対角成分はベクトル $a in RR^r$ と $b in RR^m$ で表せる．
これらの対角行列の要素は以下のように定義される．
]

$
Lambda_A_(i j) := cases(
a_i "if" i = j,
0 "else"
)
$

$
Lambda_B_(i j) := cases(
b_i "if" i = j,
0 "else"
)
$

#par()[
  この手法では，モデル内の全てのレイヤに対して同じ $A, B$ のペアを用いる．
また，学習パラメタとして $a, b$ ベクトルのみを学習対象とすることで，計算リソースを削減することが可能となる．

　この手法で特筆すべきは，$r << min(m, n)$ となる $r$ は必要ないということである．
LoRAの追加パラメタ数は $r(m + n)$ 個であるのに対し，
VeRAの追加パラメタ数は $m + r$ 個であるため，
@kopiczko2024veravectorbasedrandommatrix 中でもLoRAより大きい $r$ を設定している．
]

== QLoRA

#par()[
　QLoRA @dettmers2023qloraefficientfinetuningquantized は，LoRAの学習に量子化を適用する手法である．
事前学習モデルのパラメタは $[-1, 1]$ の変域をとり正規分布に従う性質があるため，
「k-bit NormalFloat Quantization」という正規分布する少数の情報損失を抑える量子化手法を適用する．

　QLoRAの手法では，学習時に16bitで量子化されたパラメタを用い，保存時には4bitで量子化する．
@dettmers2023qloraefficientfinetuningquantized に示される4bit量子化の分位数をTable 1に示す．
]

#figure(
  align(center)[
    #set text(size: 8pt)
    #set par(leading: 0em)
    #table(
      columns: 4,
      [-1.0                ], [0b0000],　[0.07958029955625534], [0b1000],
      [-0.6961928009986877 ], [0b0001],　[0.16093020141124725], [0b1001],
      [-0.5250730514526367 ], [0b0010],　[0.24611230194568634], [0b1010],
      [-0.39491748809814453], [0b0011],　[0.33791524171829224], [0b1011],
      [-0.28444138169288635], [0b0100],　[0.44070982933044434], [0b1100],
      [-0.18477343022823334], [0b0101],　[0.5626170039176941 ], [0b1101],
      [-0.09105003625154495], [0b0110],　[0.7229568362236023 ], [0b1110],
      [0.0                 ], [0b0111],　[1.0                ], [0b1111],
    )
  ],
  caption: "4bit量子化の分位数",
)

= LoRA系手法の比較と傾向

== 特徴の比較

#par()[
　LoRA系手法の比較をTable 2 に示す．
]

#let lora_row = {
  (
    [LoRA],
    [スタンダードなLoRA手法],
    [
      - ベーシックな実装
    ],
    [
      - $r$ の調整が必要
    ],
  )
}

#let lora_fa_row = {
  (
    [LoRA-FA],
    [
      $A$ のランダム初期化と固定
    ],
    [
      - 計算時のメモリ使用量の削減\
        (LoRAと比較して1.4倍のメモリ効率\
        @zhang2023lorafamemoryefficientlowrankadaptation 4.2 Table 4 GPU Memory Usage)

    ],
    [
      - $A$ のランダム初期化による\
        性能低下の可能性
    ],
  )
}

#let adalora_row = {
  (
    [AdaLoRA],
    [
      レイヤごとの重要度に応じた\
      $r$ 値の動的割り当て
    ],
    [
      - $r$ の調整が不要
      - モデルの性能を保ちつつ\
        学習パラメタ数を削減\
        (LoRAとの比較してほぼ全ての\
        タスクにおいて性能向上・パラメタ削減\
        @zhang2023adaloraadaptivebudgetallocation 4 Table 1 Results of DeBERTaV3-base FT)
    ],
    [
      - ランクの合計予算 $b(t)$ の設定が必要
      - 実装が複雑になること
    ],
  )
}

#let vera_row = {
  (
    [VeRA],
    [
      ベクトルによる行列の代替
    ],
    [
      - 保存するファイルサイズの大幅な削減\
        ($a, b$ベクトル，$A, B$生成用乱数シードのみ
        @kopiczko2024veravectorbasedrandommatrix 3.2 PARAMETERCOUNT)
      - パラメタの大幅な削減\
        (@kopiczko2024veravectorbasedrandommatrix 4 Table 1\~5 $r$の調整により\
        ばらつきはあるものの数倍\~数十倍の\
        パラメタ削減)
    ],
    [
      - 追加の行列演算による推論時間の増加
        (LoRAと比較して1.8%の時間増加\
        @kopiczko2024veravectorbasedrandommatrix C Table 12 Impact on GPU & Time)
    ],
  )
}

#let qlora_row = {
  (
    [QLoRA],
    [
      量子化によるパラメタ削減\
      16bitで学習，4bitで保存
    ],
    [
      - 計算時のメモリ使用量の削減\
        (65Bのモデルを41GBのメモリで学習\
        @dettmers2023qloraefficientfinetuningquantized 5.3 Table 6 Scores & Memory)
      - モデルの量子化と併用し扱えるパラメタ数が増加
      - 性能の維持\
        (@dettmers2023qloraefficientfinetuningquantized 4 Table 3 comparing to LoRA 16bit,\
        QLoRA Int8/FP4/NF4)
    ],
    [
      - 高精度を求められるタスク(数学など)\
        の性能低下リスク(@dettmers2023qloraefficientfinetuningquantized 6.1 Math)
    ],
  )
}



#place(
  top + center,
  scope: "parent",
  float: true,
  clearance: 2em,
)[
  #figure(
    [
      #set text(size: 8pt)
      #set par(leading: 0.8em)
      #table(
        columns: 4,
        align: (center, left, left, left),
        table.header([手法], [概要], [メリット], [デメリット]),
        ..lora_row,
        ..lora_fa_row,
        ..adalora_row,
        ..vera_row,
        ..qlora_row,
      )
    ],
    caption: "LoRA系手法の比較",
  )
]

== 傾向

#par()[
　ここまで取り上げてきたLoRA系の手法の目指す方向性について，
主に2つの傾向が見られる．

　1つ目は，計算リソースの削減である．
この特徴はどのLoRA手法にも共通して見られるものであり，
行列の固定 (LoRA-FA) やランクの動的調整 (AdaLoRA) ，
ベクトルによる行列の代替 (VeRA)，
量子化によるメモリ使用量の削減 (QLoRA) など，
様々な工夫がなされている．

　2つ目は，ファイルサイズの削減である．
この特徴はVeRAやQLoRAに見られるものであり，
特にベクトルと乱数シードのみの保存で済むVeRAの成果は顕著である．

　以上の「計算リソースの削減」と「ファイルサイズの削減」の2つの傾向は，
LoRA系手法の今後の研究においても重要なポイントとなると考えられる．
]

= 考察

#par()[
　本節では，ここまで取り上げたLoRAの特徴をもとに，
その実装的な有用性や今後の研究における活用について考察する．

　近年，サービスとして提供されている高性能な大規模モデル，例えばChatGPTやGemini，DeepSeekなどは，
大規模な計算リソースをもつ組織で開発されている。これまでの手法であれば，
このモデルを個別のタスクに対してFFTするのにも多大な計算リソースが必要となっていた。
しかし，LoRAを用いることで，このような大規模モデルを少ない計算リソースでFTすることが可能となる。
これにより，これまでは競争力のなかった中小企業や個人でも，
大規模モデルを活用したサービスを開発することが可能になると考えられる。

　更に最近では，機械学習モデルをエッジデバイスに搭載することが求められるようになっている。
例えば、車の自動運転などのようにリアルタイム性・正確性が求められるタスクにおいて，
クラウド上でモデルの推論を行うことは現実的ではない。
このような場合にも，使用するハードウェアの制約に合わせたモデル設計をLoRAを用いて行うことで，
スペックに制約の多いデバイス上でも、更に高性能なモデルを活用することが可能となる。

　来年度の研究では，4足歩行ロボットの制御をLLMから行うことを目指している。
このテーマでは，周囲の環境の認識，ロボットの振る舞い決定，周囲の人間とのコミュニケーションなど，
多様なタスクをLLMに適応させる選択肢が考えられる。
今回調査したLoRA系手法を活用し，用いるデバイスやタスクの難易度に応じた手法の選択を行っていきたい。
]

= 参考文献
#bibliography("refer.bib", title: none)