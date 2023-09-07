## Processing data

`datasets` library has been introduced. This can help in downloading the dataset and caching it on disk. One can also pre-process this data. (to do: find data preprocessing design patterns)

Another aspect that makes sense to know in terms of nlp applications would be to also understand what datasets are used to train different types of models.

`Glue` is a benchmark dataset that is used to train and evaluate models on many tasks. Access [this](https://gluebenchmark.com) link.


## Pre processing the data

Many nlp tasks require two sentences to be concatenated as one input. The tokenization scheme for such scenarios looks as given below

```shell
[cls] sentence one tokens [sep] sentence two tokens [sep]
```
The `tokenizer` can handle this if both the sentences are provided as positional args

```python
inputs = tokenizer(sent1,sent2)
```

## Dynamic padding

Not all inputs are going to be of same size. Hence padding is essential. There are two choices that one has:

1. Pad based on the longest input in the `data`
2. Pad based on the longest input in the `batch`

**Pros and cons**

- Choice 1
    - Pros: Can work with any hardware acceleration possible single gpu, multiple gpu or tpu.
    - Cons: Model has to do unnecessary extra work.

- Choice 2
    - Pros: Can save the model from doing unnecessary processing.
    - Cons: Doesn't work in some hardware acceleration scenarios.