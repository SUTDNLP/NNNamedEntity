NNNamedEntity
======
NNNamedEntity is a package for Named Entity Recognition using neural networks based on package LibN3L. It includes different combination of ***Neural network architectures*** (TNN, RNN, GatedNN, LSTM and GRNN) with ***Objective function***(sigmoid, CRF max-margin, CRF maximum likelihood). It also provide the combination of ***Sparse feature*** with above models. This package supports user-defined neural network structures.

Demo system
======
* Download [LibN3L](https://github.com/SUTDNLP/LibN3L) library and configure your system. Please refer to [Here](https://github.com/SUTDNLP/LibN3L)
* Open [CMakeLists.txt](CMakeLists.txt) and change " /your_directory/LibN3L/" into the directory of your [LibN3L](https://github.com/SUTDNLP/LibN3L) package.
* Run "sh demo-entity.sh".

The demo system includes English name entity recognition sample data("Entity.train", "Entity.dev" and "Entity.test"), English word embeding sample file("sena.emb") and parameter setting file("demo.option"). All of these files are gathered at folder [NNNamedEntity/example](example). 
This demo system runs a ***SparseTNNCRFMLLabeler*** model which means a traditional neural network with sparse feature and use CRF maximun likelihood as the objective function. 
After the demo system will create two files: "Entity.devOUTdemo" and "Entity.testOUTdemo" at [NNNamedEntity/example](example). These two files are the output tagged file for dev and test examples.