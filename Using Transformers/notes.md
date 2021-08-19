## Behind the pipelines
HF offers `pipelines` to use pre-trained models. If one has to make some modifications or fine tune existing models then one needs to use other slightly lower level apis. Tokenizer is one such slightly low level api.

**Tokenizer:** 
- Splitting the input into words, sub-words or symbols
- Mapping each token to an integer
- Any additional inputs that would be useful for the model such as padding etc

Every pre-trained model needs the text to be processed in a certain way hence while using a pre-trained model one will need access to the tokenizer used while training the model or something similar. 

HF provides the tokenizer for a particular model via `AutoTokenizer` class. Below snippet from the official course

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

The second slightly lower level api is that of `models`. This lets you download pre-trained models. The relevant class is `AutoModel`

The following snippet is from the official course:

```python
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```
The `Auto*` maps to different models as shown in the table below

|Suffix after `Auto`|Description|
|-------------------|-----------|
|*Model| Only hidden states from the pre-trained model no head|
| *ForCausalLM| For GPT family of models, head is retained|
|*ForMaskedLM| BERT style masked language  models, autocomplete systems etc|
|*ForMultipleChoice| BERT/Encoder based multiple choice model|
|*ForSequenceClassification| Text classification models ala BERT/GPT whole shebang|
|*ForTokenClassification| NER< POS etc, mostly BERT family of models| 