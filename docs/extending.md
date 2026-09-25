# フレームワークの拡張方法

独自の Model・Trainer・Processor・Pipeline は、[contracts.py](../src/torchmine/contracts.py) の対応する抽象基底クラスを**必ず継承**して追加します。以下はディレクトリ構成です。

```text
src/torchmine/
  contracts.py
  factory.py
  frameworks/
    registry.py
    models/
    trainers/
    processors/
  pipelines/
```

## 1. 部品の実装
部品は以下の粒度で作成します
- Model: ニューラルネットワークの構造
- Trainer: 学習方法（MSEなど）
- Processor: 学習の前処理・後処理
- Pipeline: 全体の学習の流れ

| 追加する部品 | 基底クラス | 最低限実装する操作 | 既存の例 |
| --- | --- | --- | --- |
| Model | `ModelBase` | `forward(x: Tensor) -> Tensor` | [Conv2DClassifier](../src/torchmine/frameworks/models/conv2d_classifier.py) |
| Trainer | `TrainerBase` | `train_epoch(loader) -> float`、`evaluate(loader) -> float` | [MSETrainer](../src/torchmine/frameworks/trainers/mse_trainer.py) |
| Processor | `ProcessorBase` | `transform(data: Tensor) -> Tensor` | [NormalizerProcessor](../src/torchmine/frameworks/processors/normalizer.py) |
| Pipeline | `PipelineBase` | `train_data(data)`、`predict_data(data: Tensor) -> Tensor` | [WorkflowEarlyStoppingPipeline](../src/torchmine/pipelines/workflow_early_stopping.py) |


## 2. Registry への登録

部品を実装したら、[frameworks/registry.py](../src/torchmine/frameworks/registry.py) にて実装した部品を登録してください。

```python
from .models.my_model import MyModel


def register_components(registry) -> None:
    registry.register_model("my_model", MyModel)
```


## 3. 利用する Pipeline に渡す

利用側で選んだ Pipeline 実装の設定に登録名を指定します。
