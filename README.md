NNNamedEntity
======
NNNamedEntity is a package for Named Entity Recognition using neural networks based on package LibN3L. It includes different combination of ***Neural network architectures*** (TNN, RNN, GatedNN, LSTM and GRNN) with ***Objective function***(sigmoid, CRF max-margin, CRF maximum likelihood). It also provides the capability of combination of ***Sparse feature*** with above models. In addition, this package can easily support various user-defined neural network structures.

Demo system
======
* Download [LibN3L](https://github.com/SUTDNLP/LibN3L) library and configure your system. Please refer to [Here](https://github.com/SUTDNLP/LibN3L)
* Open [CMakeLists.txt](CMakeLists.txt) and change " /your_directory/LibN3L/" into the directory of your [LibN3L](https://github.com/SUTDNLP/LibN3L) package.
* Run the [demo-entity.sh](demo-entity.sh) file: `sh demo-entity.sh`

The demo system includes English name entity recognition sample data(["Entity.train"](example/Entity.train), ["Entity.dev"](example/Entity.dev) and ["Entity.test"](example/Entity.test), English word embeding sample file(["sena.emb"](example/sena.emb) and parameter setting file(["demo.option"](example/demo.option). All of these files are gathered at folder [NNNamedEntity/example](example).
 
This demo system runs a ***SparseTNNCRFMLLabeler*** model which means a traditional neural network with sparse feature and use CRF maximun likelihood as the objective function. 

The demo system will create two files: "Entity.devOUTdemo" and "Entity.testOUTdemo" at [NNNamedEntity/example](example). These two files are the output tagged file for dev and test examples.

Feature format
======
Consider following sentence:
`Foreign - invested enterprises have played a prominent role in improving China 's export commodity structure . ` in the [Entity.test](example/Entity.test). 
The sample features for word `China` is 
`China [S]PoCNNP [S]PoBiLVBG.NNP [S]PoBiNNNP.POS [S]PoTrVBG.NNP.POS [S]WPCChina.NNP [S]UnCChina [S]UnLimproving [S]UnN's [S]CaC1 [S]CaL0 [S]CaN0 [S]CaCC1China [S]CaLC0China [S]CaNC0China [S]CaLL0improving [S]CaNN0's [S]ShC2111 [S]ShL1111 [S]ShN3133 [S]BShL11112111 [S]BShN21113133 [S]ConCnone [S]ConLnone [S]ConNnone [S]ConCaL0none [S]ConCaN0none [S]CaConL1none [S]BiLTin.improving [S]BiNT's.export [S]BiLimproving.China [S]BiNChina.'s [S]ClC301 [S]ClL448 [S]ClN181 [S]BClL448.301 [S]BClN301.181 [S]PrC0C [S]PrC1h [S]PrC2i [S]PrC3n [S]SuC0h [S]SuC1i [S]SuC2n [S]SuC3a [S]SuL0v [S]SuL1i [S]SuL2n [S]SuL3g [S]PrN0' [S]PrN1s [S]PrN2*N* [S]PrN3*N* [T]NNP B-GPE`

where

* The first word `China` means current word.
* Feature starts with `[S]` means sparse feature. For example, `[S]BiNChina.'s` can be divided as three part(`[S]+BiN+China.'s`): `[S]` represents sparse feature; `BiN` means this feature represents word bigram information for current word and next word; `China.'s` is the combination fo current word and next word. The `BiN` part can be replaced by any word except the same word in other features.
* Feature starts with "[T" are additional targets which need to be embedded. In our example, `[T]NNP` means the POS tagger information for each word will be embedded in our neural network. You can add other features which need to be embedded in following format `[T1]feature1 [T2]feature2`.
* The last tag `B-GPE` is the label for current word, it is used for model evaluation.


Monitoring information
=====
During the running of this NER system, it may print out the follow log information:
`Recall:	P=97/199=0.487437, Accuracy:	P=97/162=0.598765, Fmeasure:	0.537396`
`test:`
`Recall:	P=158/267=0.59176, Accuracy:	P=158/226=0.699115, Fmeasure:	0.640974`
`Exceeds best previous performance of 0.523161. Saving model file..`
The first "Recall..." line shows you the performance of the dev set and the second "Recall..." line shows 
you the performance of the test set.
