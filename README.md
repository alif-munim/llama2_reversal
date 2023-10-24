## Curing the Reversal Curse in Language Models
A minimal reproduction of the github repository [lukasberglund/reversal_curse](https://github.com/lukasberglund/reversal_curse/tree/main) and corresponding [paper](https://owainevans.github.io/reversal_curse.pdf) by Berglund et al. The aim of this repository is to evaluate the reversal curse phenomenon in various language model architectures and explore methods to mitigate it.

## Notes
An especially interesting paper related to this domain is ROME (Rank-One Model Editing), first described in [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) by Meng et al. It turns out fact completion in transformer-based language models can causally traced within specific hidden states, and the facts themselves can be localized in the MLP layers of the transformers. The MLP blocks work as key-value stores, with the last subject token acting as the key, and the MLP output value encoding properties about that key.

## Experiments
1. Currently playing around with bi-directional language models including BART and T5 to see if they can automatically capture bidirectional factual associations during training.
2. Reverse associations can also be manually inserted after training. [Recent work](https://twitter.com/JasonForJoy/status/1714154604366336463) proposes new editing methods that can insert new facts bi-directionally in one go.
