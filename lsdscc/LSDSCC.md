# Experiments
- Models
    * Basic seq2seq
    * Attention seq2seq
    * Greedy seq2seq
        - Li 2016 b
        - diversity rate=0.1,0.8
    * Greedy Attention
        * similar to greedy.
        * diversity rate=0.1,0.8
    * MMI decode
        - Li 2016 a
        
- Tensorflow
- Adam optimizer
- prune words whose freq are below 2.
- source vocab: 42257
- target vocab: 46865
- batch_size 50
- hidden size of encoder: 256
- hidden size of decoder: 512
- lr: 2e-4
- gradients clipping: within [-3, 3]
- train on a single GPU for at least one week.
- beam size: k = 50
- dropout 0.5

        