NNNamedEntity
======
NNNamedEntity is a package for Named Entity Recognition using neural networks based on package LibN3L. It includes different combination of ***Neural network architectures*** (TNN, RNN, GatedNN, LSTM and GRNN) with ***Final mapping functions***(sigmoid, CRF max-margin, CRF maximum likelihood). It also provide the combination of ***Sparse feature*** with above models. This package supports user-defined neural network structures.

Demo system
======
* Download [LibN3L](https://github.com/SUTDNLP/LibN3L) library and configure your system. Please refer to [Here](https://github.com/SUTDNLP/LibN3L)
* Open [CMakeLists.txt](CMakeLists.txt) and change " /your_directory/LibN3L/" into the directory of your [LibN3L](https://github.com/SUTDNLP/LibN3L) package.
* Run "sh demo-entity.sh".

The demo system includes English name entity recognition sample data("Entity.train", "Entity.dev" and "Entity.test"), English word embeding sample file("sena.emb") and parameter setting file("demo.option"). All of these file are gathered at folder /example. 