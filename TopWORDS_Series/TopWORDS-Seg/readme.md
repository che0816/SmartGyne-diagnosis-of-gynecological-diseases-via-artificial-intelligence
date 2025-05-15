# TopWORDS-Seg

This is the source code for the method TopWORDS-Seg proposed by Changzai Pan, Maosong Sun and Ke Deng in the article "TopWORDS-Seg: Simultaneous Text Segmentation and Word Discovery for Open-Domain Chinese Texts via Bayesian Inference" . This method is an extension of TopWORDS introduced in the paper "On the Unsupervised Analysis of Domain-Specific Chinese Texts" published in PNAS.

## Background

TopWORDS-Seg is an text segmentation algorithm based on Bayesian framework by integrating TopWORDS, an effective word discoverer, and a strong text segmenter(PKUSSEG or other segmenter which can be changed due to target text). TopWORDS-Seg can achieve effective text segmentation and word discovery simultaneously in open domain.

## Get Started

To get started quickly, you can quickly run TopWORDS-Seg with the following command to get a quick result: 

```bash
python main.py --inputfile_dir 'path\examples\text.txt' --punctuations_dir 'path\examples\punctuation.json' --prior_words_dir 'path\examples\prior_word.txt' --output_dir 'path\examples\output'--kappa_d 0.9 --kappa_s 0.001 --tau_l 15 --tau_f 5
```


The script can be run from the command line with the following parameters:

- `--inputfile_dir`: Path to the input text file.

  ```bash
  引言远在古希腊时期，发明家就梦想着创造能自主思考的机器。神话人物皮格马利翁、代达罗斯和赫淮斯托斯可以被看作传说中的发明家，而加拉蒂亚、塔洛斯和潘多拉则可以被视为人造生命。
  当人类第一次构思可编程计算机时，就已经在思考计算机能否变得智能尽管这距造出第一台计算机还有一百多年。如今，人工智能已经成为一个具有众多实际应用和活跃研究课题的领域，并且正在蓬勃发展。我们期望通过智能软件自动地处理常规劳动、理解语音或图像、帮助医学诊断和支持基础科学研究。
  ...
  ```


  The script can be run from the command line with the following parameters:

- `--punctuations_dir`: Path to the punctuations JSON file.

  ```bash
  [ ";", "、", "?", "。"]
  ```

- `--prior_words_dir`: Path to the prior words text file, with words and their frequency. 

  ```bash
  机器学习 100
  深度学习 100
  ```

- `--output_dir`: Output directory for the segmentation results.

- `--kappa_d`: Kappa value for parameter estimation (default: 0.5).

- `--kappa_s`: Kappa value for segmentation (default: 0.001).

- `--tau_l`: Tau value for length threshold (default: 15).

- `--tau_f`: Tau value for frequency threshold (default: 5).

### Special Case: TopWORDS Method

When both `kappa_d` and `kappa_s` are set to `1`, the TopWORDS_Seg method reverts to  the original TopWORDS method.


## Dependencies

- `argparse`， `jieba` (or other prior segmentation tool package ), `scipy`, `tqdm`, `pandas`, `nltk`
