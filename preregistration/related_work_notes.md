# Related work: historical negatives in training (checked 31 Oct 2026)

Historical negatives at training time are prior art. The report must cite these and must not present HB, HM or HX as a new idea.

- ATNSF, Jiang, Zhang, Gao, Zheng, "History Doesn't Repeat, but Its Patterns Echo", IJCAI 2026, pp. 2717-2725. https://www.ijcai.org/proceedings/2026/302
  Mixes random and historical edges in training, with historical negatives taken from before the batch and batch positives removed. Also uses a pairwise softmax loss.
  Section 6.3 runs TPNet in DyGLib, so this is the closest match to HB. Their Table 2, AUC x 100, transductive, base -> ours:
  Wiki hist 79.46 -> 90.39, Enron hist 80.73 -> 82.77, UCI hist 80.34 -> 83.35, Can. hist 85.27 -> 92.36. Random changes little, with small drops on some datasets.
  Their gamma study on UCI shows the same trade-off we see: more hard negatives help hist and ind and cost some rnd.
- CurNM, Chen, Zheng, Song, "Curriculum Negative Mining for Temporal Networks", Neural Networks 191, Dec 2025. https://arxiv.org/abs/2407.17070
  Negative pool of historical neighbours and random negatives, for JODIE, TGN and DySAT. Test negatives are half random, half historical.
- ENS, Gao, Liu, Wu, Du, Hu, "Towards a better negative sampling strategy for dynamic graphs", Neural Networks 173, May 2024. DOI 10.1016/j.neunet.2024.106175
  Built on historical dependence. Full text not read, so its test protocol is unverified.
- Poursafaei et al., NeurIPS 2022, introduced historical and inductive negatives for evaluation only. https://arxiv.org/abs/2207.10128

## What this changes
- Our "best published" column came from SNAM's table, which leaves out ATNSF. Its 77.72 for Can. Parl. is not the best published figure for this task setting once historical training is counted.
- ATNSF reports AUC and we report AP, so the numbers cannot be compared directly. They also use a pairwise loss and we keep BCE.
- What is left that is ours: TPNet with the strict sketch on the long-horizon datasets ATNSF skipped (US Legis., UN Trade, UN Vote); selection on historical validation;
  negatives restricted to pairs seen strictly before the batch; the interaction with multi-scale sketches (HM vs HB adds little); three seeds with pre-registered rules.

## Seed spread on UN Vote, historical test AP (checked 31 Oct 2026)
- TPNet paper, Table 7 (arXiv 2410.04013, 5 runs): 74.68 +/- 1.38.
- Our job 1 on the paper's code (leak present, 30 epochs, 3 seeds): 72.38, 76.01, 76.43 = 74.94 +/- 2.23. Matches the paper.
- SNAM's retraining with corrected code (5 seeds): 64.78 +/- 1.22.
- Our V3 config B (leak fixed, 100 epochs, selected on random validation): 60.25, 52.37 so far; seed 1 pending. Both seeds have validation AP 56.57, so only the historical score moved. Cause not established.
