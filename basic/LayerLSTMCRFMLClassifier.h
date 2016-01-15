/*
 * LSTMCRFMLClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_LSTMCRFMLClassifier_H_
#define SRC_LSTMCRFMLClassifier_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Feature.h"
#include "Metric.h"
#include "N3L.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//A native neural network classfier using only word embeddings
template<typename xpu>
class LSTMCRFMLClassifier {
public:
  LSTMCRFMLClassifier() {
    _dropOut = 0.5;
  }
  ~LSTMCRFMLClassifier() {

  }

public:
  LookupTable<xpu> _words;
  LookupTable<xpu> _chars;
  // tag variables
  int _tagNum;
  int _tag_outputSize;
  vector<int> _tagSize;
  vector<int> _tagDim;
  NRVec<LookupTable<xpu> > _tags;

  int _wordcontext, _wordwindow;
  int _wordSize;
  int _wordDim;

  int _charcontext, _charwindow;
  int _charSize;
  int _charDim;
  int _char_outputSize;
  int _char_inputSize;

  int _lstmhiddensize;
  int _hiddensize;
  int _inputsize, _token_representation_size;
  UniLayer<xpu> _olayer_linear;
  UniLayer<xpu> _tanh_project;
  UniLayer<xpu> _tanhchar_project;
  AttentionPooling<xpu> _gatedchar_pooling;
  LSTM<xpu> rnn_left_project;
  LSTM<xpu> rnn_right_project;
  MLCRFLoss<xpu> _crf_layer;

  int _labelSize;

  Metric _eval;

  dtype _dropOut;
// the same last variables
  LookupTable<xpu> _last_words;
  LookupTable<xpu> _last_chars;
  // tag variables
  int _last_tagNum;
  int _last_tag_outputSize;
  vector<int> _last_tagSize;
  vector<int> _last_tagDim;
  NRVec<LookupTable<xpu> > _last_tags;

  int _last_wordcontext, _last_wordwindow;
  int _last_wordSize;
  int _last_wordDim;

  int _last_charcontext, _last_charwindow;
  int _last_charSize;
  int _last_charDim;
  int _last_char_outputSize;
  int _last_char_inputSize;

  int _last_lstmhiddensize;
  int _last_hiddensize;
  int _last_inputsize, _last_token_representation_size;
  UniLayer<xpu> _last_olayer_linear;
  UniLayer<xpu> _last_tanh_project;
  UniLayer<xpu> _last_tanhchar_project;
  AttentionPooling<xpu> _last_gatedchar_pooling;
  LSTM<xpu> _lastrnn_left_project;
  LSTM<xpu> _lastrnn_right_project;
  MLCRFLoss<xpu> _last_crf_layer;

  int _last_labelSize;

  Metric _last_eval;

  dtype _last_dropOut;

  bool _secondLSTM = false;


public:

  inline void init(const NRMat<dtype>& wordEmb, int wordcontext, const NRMat<dtype>& charEmb, int charcontext, const NRVec<NRMat<dtype> >& tagEmbs, int labelSize, int charhiddensize,
      int lstmhiddensize, int hiddensize) {
    _wordcontext = wordcontext;
    _wordwindow = 2 * _wordcontext + 1;
    _wordSize = wordEmb.nrows();
    if (_secondLSTM) {
      _wordDim = _last_hiddensize;
    }
    else {
      _wordDim = wordEmb.ncols();
    }
    // tag variables
    _tagNum = tagEmbs.size();
    if (_tagNum > 0) {
      _tagSize.resize(_tagNum);
      _tagDim.resize(_tagNum);
      _tags.resize(_tagNum);
      for (int i = 0; i < _tagNum; i++){
        _tagSize[i] = tagEmbs[i].nrows();
        _tagDim[i] = tagEmbs[i].ncols();
        _tags[i].initial(tagEmbs[i]);
      }
      _tag_outputSize = _tagNum * _tagDim[0];
    }
    else {
      _tag_outputSize = 0;
    }

    _charcontext = charcontext;
    _charwindow = 2 * _charcontext + 1;
    _charSize = charEmb.nrows();
    _charDim = charEmb.ncols();
    _char_inputSize = _charwindow * _charDim;
    _char_outputSize = charhiddensize;

    _labelSize = labelSize;
    _hiddensize = hiddensize;
    _lstmhiddensize = lstmhiddensize;
    _token_representation_size = _wordDim + _char_outputSize + _tag_outputSize;
    _inputsize = _wordwindow * _token_representation_size;
    _words.initial(wordEmb);
    _chars.initial(charEmb);   

    rnn_left_project.initial(_lstmhiddensize, _inputsize, true, 20);
    rnn_right_project.initial(_lstmhiddensize, _inputsize, false, 30);
    _gatedchar_pooling.initial(_char_outputSize, _wordDim, true, 40);
    _tanhchar_project.initial(_char_outputSize, _char_inputSize, true, 50, 0);
    _tanh_project.initial(_hiddensize, 2 * _lstmhiddensize, true, 55, 0);
    _olayer_linear.initial(_labelSize, _hiddensize, false, 60, 2);

    _crf_layer.initial(_labelSize, 70);

  }


  inline void initlast(const NRMat<dtype>& wordEmb, int wordcontext, const NRMat<dtype>& charEmb, int charcontext, const NRVec<NRMat<dtype> >& tagEmbs, int labelSize, int charhiddensize,
      int lstmhiddensize, int hiddensize) {
    _last_wordcontext = wordcontext;
    _last_wordwindow = 2 * _wordcontext + 1;
    _last_wordSize = wordEmb.nrows();
    _last_wordDim = wordEmb.ncols();

    // tag variables
    _last_tagNum = tagEmbs.size();
    if (_last_tagNum > 0) {
      _last_tagSize.resize(_last_tagNum);
      _last_tagDim.resize(_last_tagNum);
      _last_tags.resize(_last_tagNum);
      for (int i = 0; i < _last_tagNum; i++){
        _last_tagSize[i] = tagEmbs[i].nrows();
        _last_tagDim[i] = tagEmbs[i].ncols();
        _last_tags[i].initial(tagEmbs[i]);
      }
      _last_tag_outputSize = _last_tagNum * _last_tagDim[0];
    }
    else {
      _last_tag_outputSize = 0;
    }

    _last_charcontext = charcontext;
    _last_charwindow = 2 * _last_charcontext + 1;
    _last_charSize = charEmb.nrows();
    _last_charDim = charEmb.ncols();
    _last_char_inputSize = _last_charwindow * _last_charDim;
    _last_char_outputSize = charhiddensize;

    _last_labelSize = labelSize;
    _last_hiddensize = hiddensize;
    _last_lstmhiddensize = lstmhiddensize;
    _last_token_representation_size = _last_wordDim + _last_char_outputSize + _last_tag_outputSize;
    _last_inputsize = _last_wordwindow * _last_token_representation_size;
    _last_words.initial(wordEmb);
    _last_chars.initial(charEmb);   

    _lastrnn_left_project.initial(_last_lstmhiddensize, _last_inputsize, true, 20);
    _lastrnn_right_project.initial(_last_lstmhiddensize, _last_inputsize, false, 30);
    _last_gatedchar_pooling.initial(_last_char_outputSize, _last_wordDim, true, 40);
    _last_tanhchar_project.initial(_last_char_outputSize, _last_char_inputSize, true, 50, 0);
    _last_tanh_project.initial(_last_hiddensize, 2 * _last_lstmhiddensize, true, 55, 0);
    _last_olayer_linear.initial(_last_labelSize, _last_hiddensize, false, 60, 2);

    _last_crf_layer.initial(_last_labelSize, 70);

  }


  inline void release() {
    _words.release();
    _chars.release();
    // add tags release
    if (_tagNum > 0) {
      for (int i = 0; i < _tagNum; i++){
        _tags[i].release();
      }
    }
    _olayer_linear.release();
    _tanh_project.release();
    _tanhchar_project.release();
    _gatedchar_pooling.release();
    rnn_left_project.release();
    rnn_right_project.release();
    _crf_layer.release();

  }

  inline dtype process(const vector<Example>& examples, int iter) {
    _eval.reset();

    int example_num = examples.size();
    dtype cost = 0.0;
    int offset = 0;
    for (int count = 0; count < example_num; count++) {
      const Example& example = examples[count];

      int seq_size = example.m_features.size();

      vector<Tensor<xpu, 2, dtype> > input(seq_size), inputLoss(seq_size);
      vector<Tensor<xpu, 2, dtype> > lstmoutput(seq_size), lstmoutputLoss(seq_size);
      vector<Tensor<xpu, 2, dtype> > i_project_left(seq_size), o_project_left(seq_size), f_project_left(seq_size);
      vector<Tensor<xpu, 2, dtype> > mc_project_left(seq_size), c_project_left(seq_size), my_project_left(seq_size);
      vector<Tensor<xpu, 2, dtype> > i_project_right(seq_size), o_project_right(seq_size), f_project_right(seq_size);
      vector<Tensor<xpu, 2, dtype> > mc_project_right(seq_size), c_project_right(seq_size), my_project_right(seq_size);
      vector<Tensor<xpu, 2, dtype> > project_left(seq_size), project_leftLoss(seq_size), project_right(seq_size), project_rightLoss(seq_size);
      vector<Tensor<xpu, 2, dtype> > project(seq_size), projectLoss(seq_size);
      vector<Tensor<xpu, 2, dtype> > output(seq_size), outputLoss(seq_size);

      vector<Tensor<xpu, 3, dtype> > charprime(seq_size), charprimeLoss(seq_size), charprimeMask(seq_size);
      vector<Tensor<xpu, 3, dtype> > charinput(seq_size), charinputLoss(seq_size);
      vector<Tensor<xpu, 3, dtype> > charhidden(seq_size), charhiddenLoss(seq_size);
      vector<Tensor<xpu, 3, dtype> > chargatedpoolIndex(seq_size), chargateweight(seq_size), chargateweightMiddle(seq_size);
      vector<Tensor<xpu, 2, dtype> > chargatedpool(seq_size), chargatedpoolLoss(seq_size);
      vector<Tensor<xpu, 2, dtype> > chargateweightsum(seq_size);
      // tag number
      vector<Tensor<xpu, 3, dtype> > tagprime(seq_size), tagprimeLoss(seq_size), tagprimeMask(seq_size);
      vector<Tensor<xpu, 2, dtype> > tagoutput(seq_size), tagoutputLoss(seq_size);
      vector<Tensor<xpu, 2, dtype> > wordprime(seq_size), wordprimeLoss(seq_size), wordprimeMask(seq_size);
      vector<Tensor<xpu, 2, dtype> > wordrepresent(seq_size), wordrepresentLoss(seq_size);

      //initialize
      for (int idx = 0; idx < seq_size; idx++) {
        const Feature& feature = example.m_features[idx];

        int char_num = feature.chars.size();
        charprime[idx] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), d_zero);
        charprimeLoss[idx] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), d_zero);
        charprimeMask[idx] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), d_one);
        charinput[idx] = NewTensor<xpu>(Shape3(char_num, 1, _char_inputSize), d_zero);
        charinputLoss[idx] = NewTensor<xpu>(Shape3(char_num, 1, _char_inputSize), d_zero);
        charhidden[idx] = NewTensor<xpu>(Shape3(char_num, 1, _char_outputSize), d_zero);
        charhiddenLoss[idx] = NewTensor<xpu>(Shape3(char_num, 1, _char_outputSize), d_zero);

        chargatedpoolIndex[idx] = NewTensor<xpu>(Shape3(char_num, 1, _char_outputSize), d_zero);
        chargateweightMiddle[idx] = NewTensor<xpu>(Shape3(char_num, 1, _char_outputSize), d_zero);
        chargateweight[idx] = NewTensor<xpu>(Shape3(char_num, 1, _char_outputSize), d_zero);
        chargateweightsum[idx] = NewTensor<xpu>(Shape2(1, _char_outputSize), d_zero);
        chargatedpool[idx] = NewTensor<xpu>(Shape2(1, _char_outputSize), d_zero);
        chargatedpoolLoss[idx] = NewTensor<xpu>(Shape2(1, _char_outputSize), d_zero);

        // tag prime init
        if (_tagNum > 0) {
          tagprime[idx] = NewTensor<xpu>(Shape3(_tagNum, 1, _tagDim[0]), d_zero);
          tagprimeLoss[idx] = NewTensor<xpu>(Shape3(_tagNum, 1, _tagDim[0]), d_zero);
          tagprimeMask[idx] = NewTensor<xpu>(Shape3(_tagNum, 1, _tagDim[0]), d_one);
          tagoutput[idx] = NewTensor<xpu>(Shape2(1, _tag_outputSize), d_zero);
          tagoutputLoss[idx] = NewTensor<xpu>(Shape2(1, _tag_outputSize), d_zero);
        }
        wordprime[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
        wordprimeLoss[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
        wordprimeMask[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_one);
        
        wordrepresent[idx] = NewTensor<xpu>(Shape2(1, _token_representation_size), d_zero);
        wordrepresentLoss[idx] = NewTensor<xpu>(Shape2(1, _token_representation_size), d_zero);
        input[idx] = NewTensor<xpu>(Shape2(1, _inputsize), d_zero);
        inputLoss[idx] = NewTensor<xpu>(Shape2(1, _inputsize), d_zero);

        i_project_left[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
        o_project_left[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
        f_project_left[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
        mc_project_left[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
        c_project_left[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
        my_project_left[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
        project_left[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
        project_leftLoss[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);

        i_project_right[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
        o_project_right[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
        f_project_right[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
        mc_project_right[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
        c_project_right[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
        my_project_right[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
        project_right[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
        project_rightLoss[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);

        lstmoutput[idx] = NewTensor<xpu>(Shape2(1, 2 * _lstmhiddensize), d_zero);
        lstmoutputLoss[idx] = NewTensor<xpu>(Shape2(1, 2 * _lstmhiddensize), d_zero);        
        project[idx] = NewTensor<xpu>(Shape2(1, _hiddensize), d_zero);
        projectLoss[idx] = NewTensor<xpu>(Shape2(1, _hiddensize), d_zero);


        output[idx] = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
        outputLoss[idx] = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
      }

      //forward propagation
      //input setting, and linear setting

      if (_secondLSTM) {
        if (!formatCheck())
          return -1;
        computerHiddenScore(example.m_features, wordprime);
      }

      for (int idx = 0; idx < seq_size; idx++) {
        const Feature& feature = example.m_features[idx];
        //linear features should not be dropped out

        srand(iter * example_num + count * seq_size + idx);

        const vector<int>& words = feature.words;
        if (!_secondLSTM)
          _words.GetEmb(words[0], wordprime[idx]);

        dropoutcol(wordprimeMask[idx], _dropOut);
        wordprime[idx] = wordprime[idx] * wordprimeMask[idx];

        const vector<int>& chars = feature.chars;
        int char_num = chars.size();

        //charprime
        for (int idy = 0; idy < char_num; idy++) {
          // cout << "get chars ?" << endl;
          _chars.GetEmb(chars[idy], charprime[idx][idy]);
        }

        //char dropout
        for (int idy = 0; idy < char_num; idy++) {
          dropoutcol(charprimeMask[idx][idy], _dropOut);
          charprime[idx][idy] = charprime[idx][idy] * charprimeMask[idx][idy];
        }
      
        // char context
        windowlized(charprime[idx], charinput[idx], _charcontext);

        // char combination
        _tanhchar_project.ComputeForwardScore(charinput[idx], charhidden[idx]);

        // char gated pooling
        _gatedchar_pooling.ComputeForwardScore(charhidden[idx], wordprime[idx], chargateweightMiddle[idx], chargateweight[idx], chargateweightsum[idx], chargatedpoolIndex[idx], chargatedpool[idx]);
        // tag prime get 
        if (_tagNum > 0) {
          const vector<int>& tags = feature.tags;
          for (int idy = 0; idy < _tagNum; idy++) {
            _tags[idy].GetEmb(tags[idy], tagprime[idx][idy]);
          }
          // tag drop out
          for (int idy = 0; idy < _tagNum; idy++) {
            dropoutcol(tagprimeMask[idx][idy], _dropOut);
            tagprime[idx][idy] = tagprime[idx][idy] * tagprimeMask[idx][idy];
          }
          concat(tagprime[idx], tagoutput[idx]);
        }
      }
      // concat tag input
      if (_tagNum > 0) {
        for (int idx = 0; idx < seq_size; idx++) {
          concat(wordprime[idx], chargatedpool[idx], tagoutput[idx], wordrepresent[idx]);
        }
      }
      else {
        for (int idx = 0; idx < seq_size; idx++) {
          concat(wordprime[idx], chargatedpool[idx], wordrepresent[idx]);
        }        
      }

      windowlized(wordrepresent, input, _wordcontext);

      rnn_left_project.ComputeForwardScore(input, i_project_left, o_project_left, f_project_left, mc_project_left,
          c_project_left, my_project_left, project_left);
      rnn_right_project.ComputeForwardScore(input, i_project_right, o_project_right, f_project_right, mc_project_right,
          c_project_right, my_project_right, project_right);

      for (int idx = 0; idx < seq_size; idx++) {
        concat(project_left[idx], project_right[idx], lstmoutput[idx]);
      }
      _tanh_project.ComputeForwardScore(lstmoutput, project);



      _olayer_linear.ComputeForwardScore(project, output);

      // get delta for each output
      cost += _crf_layer.loss(output, example.m_labels, outputLoss, _eval, example_num);

      // loss backward propagation
      // output
      _olayer_linear.ComputeBackwardLoss(project, output, outputLoss, projectLoss);

      _tanh_project.ComputeBackwardLoss(lstmoutput, project, projectLoss, lstmoutputLoss);
      for (int idx = 0; idx < seq_size; idx++) {
        unconcat(project_leftLoss[idx], project_rightLoss[idx], lstmoutputLoss[idx]);
      }



      // word combination
      rnn_left_project.ComputeBackwardLoss(input, i_project_left, o_project_left, f_project_left, mc_project_left,
          c_project_left, my_project_left, project_left, project_leftLoss, inputLoss);
      rnn_right_project.ComputeBackwardLoss(input, i_project_right, o_project_right, f_project_right, mc_project_right,
          c_project_right, my_project_right, project_right, project_rightLoss, inputLoss);

      // word context
      windowlized_backward(wordrepresentLoss, inputLoss, _wordcontext);

      // decompose loss with tagoutputLoss
      if (_tagNum > 0) {
        for (int idx = 0; idx < seq_size; idx++) {
          unconcat(wordprimeLoss[idx], chargatedpoolLoss[idx], tagoutputLoss[idx], wordrepresentLoss[idx]);
          // tag prime loss
          unconcat(tagprimeLoss[idx], tagoutputLoss[idx]);
        }
      }
      else {
        for (int idx = 0; idx < seq_size; idx++) {
          unconcat(wordprimeLoss[idx], chargatedpoolLoss[idx], wordrepresentLoss[idx]);
        }        
      }

      for (int idx = 0; idx < seq_size; idx++) {
        _gatedchar_pooling.ComputeBackwardLoss(charhidden[idx], wordprime[idx], chargateweightMiddle[idx], chargateweight[idx], chargateweightsum[idx],
            chargatedpoolIndex[idx], chargatedpool[idx], chargatedpoolLoss[idx], charhiddenLoss[idx], wordprimeLoss[idx]);

        //char convolution
        _tanhchar_project.ComputeBackwardLoss(charinput[idx], charhidden[idx], charhiddenLoss[idx], charinputLoss[idx]);

        //char context
        windowlized_backward(charprimeLoss[idx], charinputLoss[idx], _charcontext);
      }

      // word fine tune
      if (_words.bEmbFineTune()) {
        for (int idx = 0; idx < seq_size; idx++) {
          const Feature& feature = example.m_features[idx];
          const vector<int>& words = feature.words;
          wordprimeLoss[idx] = wordprimeLoss[idx] * wordprimeMask[idx];
          _words.EmbLoss(words[0], wordprimeLoss[idx]);
        }
      }
      //tag fine tune
      if (_tagNum > 0) {
        for (int idy = 0; idy < _tagNum; idy++){
          if (_tags[idy].bEmbFineTune()) {
            for (int idx = 0; idx < seq_size; idx++) {
              const Feature& feature = example.m_features[idx];
              const vector<int>& tags = feature.tags;
              tagprimeLoss[idx][idy] = tagprimeLoss[idx][idy] * tagprimeMask[idx][idy];
              _tags[idy].EmbLoss(tags[idy], tagprimeLoss[idx][idy]);
            }
          }
        }
      }

      //char finetune
      if (_chars.bEmbFineTune()) {
        for (int idx = 0; idx < seq_size; idx++) {
          const Feature& feature = example.m_features[idx];
          const vector<int>& chars = feature.chars;
          int char_num = chars.size();
          for (int idy = 0; idy < char_num; idy++) {
            charprimeLoss[idx][idy] = charprimeLoss[idx][idy] * charprimeMask[idx][idy];
            _chars.EmbLoss(chars[idy], charprimeLoss[idx][idy]);
          }
        }
      }

      //release
      // std::cout << _eval.getAccuracy() << std::endl;
      // std::cout << "before release space , line 435 " << std::endl;
      for (int idx = 0; idx < seq_size; idx++) {
        FreeSpace(&(charprime[idx]));
        FreeSpace(&(charprimeLoss[idx]));
        FreeSpace(&(charprimeMask[idx]));
        FreeSpace(&(charinput[idx]));
        FreeSpace(&(charinputLoss[idx]));
        FreeSpace(&(charhidden[idx]));
        FreeSpace(&(charhiddenLoss[idx]));
        FreeSpace(&(chargatedpoolIndex[idx]));
        FreeSpace(&(chargateweightMiddle[idx]));
        FreeSpace(&(chargateweight[idx]));
        FreeSpace(&(chargateweightsum[idx]));
        FreeSpace(&(chargatedpool[idx]));
        FreeSpace(&(chargatedpoolLoss[idx]));
        
        // tag freespace
        if (_tagNum > 0) {
          FreeSpace(&(tagprime[idx]));
          FreeSpace(&(tagprimeLoss[idx]));
          FreeSpace(&(tagprimeMask[idx]));
          FreeSpace(&(tagoutput[idx]));
          FreeSpace(&(tagoutputLoss[idx]));
        }
        FreeSpace(&(wordprime[idx]));
        FreeSpace(&(wordprimeLoss[idx]));
        FreeSpace(&(wordprimeMask[idx]));
        FreeSpace(&(wordrepresent[idx]));
        FreeSpace(&(wordrepresentLoss[idx]));

        FreeSpace(&(input[idx]));
        FreeSpace(&(inputLoss[idx]));

        FreeSpace(&(i_project_left[idx]));
        FreeSpace(&(o_project_left[idx]));
        FreeSpace(&(f_project_left[idx]));
        FreeSpace(&(mc_project_left[idx]));
        FreeSpace(&(c_project_left[idx]));
        FreeSpace(&(my_project_left[idx]));
        FreeSpace(&(project_left[idx]));
        FreeSpace(&(project_leftLoss[idx]));

        FreeSpace(&(i_project_right[idx]));
        FreeSpace(&(o_project_right[idx]));
        FreeSpace(&(f_project_right[idx]));
        FreeSpace(&(mc_project_right[idx]));
        FreeSpace(&(c_project_right[idx]));
        FreeSpace(&(my_project_right[idx]));
        FreeSpace(&(project_right[idx]));
        FreeSpace(&(project_rightLoss[idx]));

        FreeSpace(&(lstmoutput[idx]));
        FreeSpace(&(project[idx]));
        FreeSpace(&(projectLoss[idx]));
        FreeSpace(&(lstmoutputLoss[idx]));
        FreeSpace(&(output[idx]));
        FreeSpace(&(outputLoss[idx]));
      }
    }

    if (_eval.getAccuracy() < 0) {
      std::cout << "strange" << std::endl;
    }

    return cost;
  }

  void predict(const vector<Feature>& features, vector<int>& results) {
    int seq_size = features.size();
    int offset = 0;

    vector<Tensor<xpu, 2, dtype> > input(seq_size);
    vector<Tensor<xpu, 2, dtype> > lstmoutput(seq_size);

    vector<Tensor<xpu, 2, dtype> > i_project_left(seq_size), o_project_left(seq_size), f_project_left(seq_size);
    vector<Tensor<xpu, 2, dtype> > mc_project_left(seq_size), c_project_left(seq_size), my_project_left(seq_size);
    vector<Tensor<xpu, 2, dtype> > i_project_right(seq_size), o_project_right(seq_size), f_project_right(seq_size);
    vector<Tensor<xpu, 2, dtype> > mc_project_right(seq_size), c_project_right(seq_size), my_project_right(seq_size);
    vector<Tensor<xpu, 2, dtype> > project_left(seq_size), project_right(seq_size);

    vector<Tensor<xpu, 2, dtype> > project(seq_size);
    vector<Tensor<xpu, 2, dtype> > output(seq_size);

    vector<Tensor<xpu, 3, dtype> > charprime(seq_size);
    vector<Tensor<xpu, 3, dtype> > charinput(seq_size);
    vector<Tensor<xpu, 3, dtype> > charhidden(seq_size);
    vector<Tensor<xpu, 3, dtype> > chargatedpoolIndex(seq_size);
    vector<Tensor<xpu, 2, dtype> > chargatedpool(seq_size);
    vector<Tensor<xpu, 3, dtype> > chargateweightMiddle(seq_size);
    vector<Tensor<xpu, 3, dtype> > chargateweight(seq_size);
    vector<Tensor<xpu, 2, dtype> > chargateweightsum(seq_size);

    vector<Tensor<xpu, 3, dtype> > tagprime(seq_size);
    vector<Tensor<xpu, 2, dtype> > tagoutput(seq_size);

    vector<Tensor<xpu, 2, dtype> > wordprime(seq_size);
    vector<Tensor<xpu, 2, dtype> > wordrepresent(seq_size);

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = features[idx];

      int char_num = feature.chars.size();
      charprime[idx] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), d_zero);
      charinput[idx] = NewTensor<xpu>(Shape3(char_num, 1, _char_inputSize), d_zero);
      charhidden[idx] = NewTensor<xpu>(Shape3(char_num, 1, _char_outputSize), d_zero);
      chargatedpoolIndex[idx] = NewTensor<xpu>(Shape3(char_num, 1, _char_outputSize), d_zero);
      chargatedpool[idx] = NewTensor<xpu>(Shape2(1, _char_outputSize), d_zero);
      chargateweightMiddle[idx] = NewTensor<xpu>(Shape3(char_num, 1, _char_outputSize), d_zero);
      chargateweight[idx] = NewTensor<xpu>(Shape3(char_num, 1, _char_outputSize), d_zero);
      chargateweightsum[idx] = NewTensor<xpu>(Shape2(1, _char_outputSize), d_zero);
      if (_tagNum > 0) {
        tagprime[idx] = NewTensor<xpu>(Shape3(_tagNum, 1, _tagDim[0]), d_zero);
        tagoutput[idx] = NewTensor<xpu>(Shape2(1, _tag_outputSize), d_zero);
      }

      wordprime[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
      wordrepresent[idx] = NewTensor<xpu>(Shape2(1, _token_representation_size), d_zero);
      input[idx] = NewTensor<xpu>(Shape2(1, _inputsize), d_zero);

      i_project_left[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      o_project_left[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      f_project_left[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      mc_project_left[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      c_project_left[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      my_project_left[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      project_left[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);

      i_project_right[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      o_project_right[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      f_project_right[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      mc_project_right[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      c_project_right[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      my_project_right[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      project_right[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);

      lstmoutput[idx] = NewTensor<xpu>(Shape2(1, 2 * _lstmhiddensize), d_zero);
      project[idx] = NewTensor<xpu>(Shape2(1, _hiddensize), d_zero);
      output[idx] = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
    }

    //forward propagation
    //input setting, and linear setting
    if (_secondLSTM)
      computerHiddenScore(features, wordprime);
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = features[idx];
      //linear features should not be dropped out

      const vector<int>& words = feature.words;
      if (!_secondLSTM)
        _words.GetEmb(words[0], wordprime[idx]);

      const vector<int>& chars = feature.chars;
      int char_num = chars.size();

      //charprime
      for (int idy = 0; idy < char_num; idy++) {
        _chars.GetEmb(chars[idy], charprime[idx][idy]);
      }

      // char context
      windowlized(charprime[idx], charinput[idx], _charcontext);

      // char combination
      _tanhchar_project.ComputeForwardScore(charinput[idx], charhidden[idx]);

      // char gated pooling
      _gatedchar_pooling.ComputeForwardScore(charhidden[idx], wordprime[idx], chargateweightMiddle[idx], chargateweight[idx], chargateweightsum[idx], chargatedpoolIndex[idx], chargatedpool[idx]);
      // tag prime get
      if (_tagNum > 0) {
        const vector<int>& tags = feature.tags;
        for (int idy = 0; idy < _tagNum; idy++){
          _tags[idy].GetEmb(tags[idy], tagprime[idx][idy]);
        }
        concat(tagprime[idx], tagoutput[idx]); 
      } 
    }
    if (_tagNum > 0) {
      for (int idx = 0; idx < seq_size; idx++) {
        concat(wordprime[idx], chargatedpool[idx], tagoutput[idx], wordrepresent[idx]);
      }
    }
    else {
      for (int idx = 0; idx < seq_size; idx++) {
        concat(wordprime[idx], chargatedpool[idx], wordrepresent[idx]);
      }      
    }

    windowlized(wordrepresent, input, _wordcontext);

    rnn_left_project.ComputeForwardScore(input, i_project_left, o_project_left, f_project_left, mc_project_left,
        c_project_left, my_project_left, project_left);
    rnn_right_project.ComputeForwardScore(input, i_project_right, o_project_right, f_project_right, mc_project_right,
        c_project_right, my_project_right, project_right);

    for (int idx = 0; idx < seq_size; idx++) {
      concat(project_left[idx], project_right[idx], lstmoutput[idx]);
    }
    _tanh_project.ComputeForwardScore(lstmoutput, project);

    _olayer_linear.ComputeForwardScore(project, output);

    // decode algorithm
    _crf_layer.predict(output, results);

    //release
    for (int idx = 0; idx < seq_size; idx++) {
      FreeSpace(&(charprime[idx]));
      FreeSpace(&(charinput[idx]));
      FreeSpace(&(charhidden[idx]));
      FreeSpace(&(chargatedpoolIndex[idx]));
      FreeSpace(&(chargatedpool[idx]));
      FreeSpace(&(chargateweightMiddle[idx]));
      FreeSpace(&(chargateweight[idx]));
      FreeSpace(&(chargateweightsum[idx]));
      if (_tagNum > 0) {
        FreeSpace(&(tagprime[idx]));
        FreeSpace(&(tagoutput[idx]));
      }
      FreeSpace(&(wordprime[idx]));
      FreeSpace(&(wordrepresent[idx]));
      FreeSpace(&(input[idx]));

      FreeSpace(&(i_project_left[idx]));
      FreeSpace(&(o_project_left[idx]));
      FreeSpace(&(f_project_left[idx]));
      FreeSpace(&(mc_project_left[idx]));
      FreeSpace(&(c_project_left[idx]));
      FreeSpace(&(my_project_left[idx]));
      FreeSpace(&(project_left[idx]));

      FreeSpace(&(i_project_right[idx]));
      FreeSpace(&(o_project_right[idx]));
      FreeSpace(&(f_project_right[idx]));
      FreeSpace(&(mc_project_right[idx]));
      FreeSpace(&(c_project_right[idx]));
      FreeSpace(&(my_project_right[idx]));
      FreeSpace(&(project_right[idx]));

      FreeSpace(&(lstmoutput[idx]));
      FreeSpace(&(project[idx]));
      FreeSpace(&(output[idx]));
    }
  }

  dtype computeScore(const Example& example) {
    int seq_size = example.m_features.size();
    int offset = 0;

    vector<Tensor<xpu, 2, dtype> > input(seq_size);
    vector<Tensor<xpu, 2, dtype> > lstmoutput(seq_size);

    vector<Tensor<xpu, 2, dtype> > i_project_left(seq_size), o_project_left(seq_size), f_project_left(seq_size);
    vector<Tensor<xpu, 2, dtype> > mc_project_left(seq_size), c_project_left(seq_size), my_project_left(seq_size);
    vector<Tensor<xpu, 2, dtype> > i_project_right(seq_size), o_project_right(seq_size), f_project_right(seq_size);
    vector<Tensor<xpu, 2, dtype> > mc_project_right(seq_size), c_project_right(seq_size), my_project_right(seq_size);
    vector<Tensor<xpu, 2, dtype> > project_left(seq_size), project_right(seq_size);

    vector<Tensor<xpu, 2, dtype> > project(seq_size);
    vector<Tensor<xpu, 2, dtype> > output(seq_size);

    vector<Tensor<xpu, 3, dtype> > charprime(seq_size);
    vector<Tensor<xpu, 3, dtype> > charinput(seq_size);
    vector<Tensor<xpu, 3, dtype> > charhidden(seq_size);
    vector<Tensor<xpu, 3, dtype> > chargatedpoolIndex(seq_size);
    vector<Tensor<xpu, 2, dtype> > chargatedpool(seq_size);
    vector<Tensor<xpu, 3, dtype> > chargateweightMiddle(seq_size);
    vector<Tensor<xpu, 3, dtype> > chargateweight(seq_size);
    vector<Tensor<xpu, 2, dtype> > chargateweightsum(seq_size);

    vector<Tensor<xpu, 3, dtype> > tagprime(seq_size);
    vector<Tensor<xpu, 2, dtype> > tagoutput(seq_size);
    vector<Tensor<xpu, 2, dtype> > wordprime(seq_size);
    vector<Tensor<xpu, 2, dtype> > wordrepresent(seq_size);

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = example.m_features[idx];

      int char_num = feature.chars.size();
      charprime[idx] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), d_zero);
      charinput[idx] = NewTensor<xpu>(Shape3(char_num, 1, _char_inputSize), d_zero);
      charhidden[idx] = NewTensor<xpu>(Shape3(char_num, 1, _char_outputSize), d_zero);
      chargatedpoolIndex[idx] = NewTensor<xpu>(Shape3(char_num, 1, _char_outputSize), d_zero);
      chargatedpool[idx] = NewTensor<xpu>(Shape2(1, _char_outputSize), d_zero);
      chargateweightMiddle[idx] = NewTensor<xpu>(Shape3(char_num, 1, _char_outputSize), d_zero);
      chargateweight[idx] = NewTensor<xpu>(Shape3(char_num, 1, _char_outputSize), d_zero);
      chargateweightsum[idx] = NewTensor<xpu>(Shape2(1, _char_outputSize), d_zero);
      if (_tagNum > 0) {
        tagprime[idx] = NewTensor<xpu>(Shape3(_tagNum, 1, _tagDim[0]), d_zero);
        tagoutput[idx] = NewTensor<xpu>(Shape2(1, _tag_outputSize), d_zero);
      }

      wordprime[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
      wordrepresent[idx] = NewTensor<xpu>(Shape2(1, _token_representation_size), d_zero);

      input[idx] = NewTensor<xpu>(Shape2(1, _inputsize), d_zero);

      i_project_left[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      o_project_left[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      f_project_left[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      mc_project_left[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      c_project_left[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      my_project_left[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      project_left[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);

      i_project_right[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      o_project_right[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      f_project_right[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      mc_project_right[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      c_project_right[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      my_project_right[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);
      project_right[idx] = NewTensor<xpu>(Shape2(1, _lstmhiddensize), d_zero);

      lstmoutput[idx] = NewTensor<xpu>(Shape2(1, 2 * _lstmhiddensize), d_zero);
      project[idx] = NewTensor<xpu>(Shape2(1, _hiddensize), d_zero);
      output[idx] = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
    }

    //forward propagation
    //input setting, and linear setting
    if (_secondLSTM) {
      // cout << "comput second score " << endl;
      computerHiddenScore(example.m_features, wordprime);
    }
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = example.m_features[idx];
      //linear features should not be dropped out

      const vector<int>& words = feature.words;
      if (!_secondLSTM)
        _words.GetEmb(words[0], wordprime[idx]);

      const vector<int>& chars = feature.chars;
      int char_num = chars.size();

      //charprime
      for (int idy = 0; idy < char_num; idy++) {
        _chars.GetEmb(chars[idy], charprime[idx][idy]);
      }

      // char context
      windowlized(charprime[idx], charinput[idx], _charcontext);

      // char combination
      _tanhchar_project.ComputeForwardScore(charinput[idx], charhidden[idx]);

      // char gated pooling
      _gatedchar_pooling.ComputeForwardScore(charhidden[idx], wordprime[idx], chargateweightMiddle[idx], chargateweight[idx], chargateweightsum[idx], chargatedpoolIndex[idx], chargatedpool[idx]);
      // tag prime get
      if (_tagNum > 0) {
        const vector<int>& tags = feature.tags;
        for (int idy = 0; idy < _tagNum; idy++){
          _tags[idy].GetEmb(tags[idy], tagprime[idx][idy]);
        }
        concat(tagprime[idx], tagoutput[idx]);  
      }
    }
    if (_tagNum > 0) {
      for (int idx = 0; idx < seq_size; idx++) {
        concat(wordprime[idx], chargatedpool[idx], tagoutput[idx], wordrepresent[idx]);
      }
    }
    else {
      for (int idx = 0; idx < seq_size; idx++) {
        concat(wordprime[idx], chargatedpool[idx], wordrepresent[idx]);
      }      
    }
    windowlized(wordrepresent, input, _wordcontext);

    rnn_left_project.ComputeForwardScore(input, i_project_left, o_project_left, f_project_left, mc_project_left,
        c_project_left, my_project_left, project_left);
    rnn_right_project.ComputeForwardScore(input, i_project_right, o_project_right, f_project_right, mc_project_right,
        c_project_right, my_project_right, project_right);

    for (int idx = 0; idx < seq_size; idx++) {
      concat(project_left[idx], project_right[idx], lstmoutput[idx]);
    }
    _tanh_project.ComputeForwardScore(lstmoutput, project);

    _olayer_linear.ComputeForwardScore(project, output);

    // get delta for each output
    dtype cost = _crf_layer.cost(output, example.m_labels);

    //release

    for (int idx = 0; idx < seq_size; idx++) {
      FreeSpace(&(charprime[idx]));
      FreeSpace(&(charinput[idx]));
      FreeSpace(&(charhidden[idx]));
      FreeSpace(&(chargatedpoolIndex[idx]));
      FreeSpace(&(chargatedpool[idx]));
      FreeSpace(&(chargateweightMiddle[idx]));
      FreeSpace(&(chargateweight[idx]));
      FreeSpace(&(chargateweightsum[idx]));
      if (_tagNum > 0) {
        FreeSpace(&(tagprime[idx]));
        FreeSpace(&(tagoutput[idx]));
      }
      FreeSpace(&(wordprime[idx]));
      FreeSpace(&(wordrepresent[idx]));
      FreeSpace(&(input[idx]));
      
      FreeSpace(&(i_project_left[idx]));
      FreeSpace(&(o_project_left[idx]));
      FreeSpace(&(f_project_left[idx]));
      FreeSpace(&(mc_project_left[idx]));
      FreeSpace(&(c_project_left[idx]));
      FreeSpace(&(my_project_left[idx]));
      FreeSpace(&(project_left[idx]));
      
      FreeSpace(&(i_project_right[idx]));
      FreeSpace(&(o_project_right[idx]));
      FreeSpace(&(f_project_right[idx]));
      FreeSpace(&(mc_project_right[idx]));
      FreeSpace(&(c_project_right[idx]));
      FreeSpace(&(my_project_right[idx]));
      FreeSpace(&(project_right[idx]));

      FreeSpace(&(lstmoutput[idx]));
      FreeSpace(&(project[idx]));
      FreeSpace(&(output[idx]));
    }

    return cost;
  }

  void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
    rnn_left_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    rnn_right_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _tanhchar_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _gatedchar_pooling.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _tanh_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _crf_layer.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _words.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _chars.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    if (_tagNum > 0) {
      for (int i = 0; i < _tagNum; i++){
        _tags[i].updateAdaGrad(nnRegular, adaAlpha, adaEps);
      }
    }
    
  }


  bool formatCheck() {
    if (_last_wordDim != _wordDim) {
      std::cout << "Word dimension mismatch, end calculation! Original dim: " << _last_wordDim << "New input dim: " << _wordDim << std::endl;
      return false;
    }
    if (_last_charDim != _charDim) {
      std::cout << "Char dimension mismatch, end calculation! Original dim: " << _last_charDim << "New input dim: " << _charDim << std::endl;
      return false;
    }
    // if (_last_tagNum != _tagNum) {
    //   std::cout << "Tag number mismatch, end calculation! Original Num: " << _last_tagNum << "New input Num: " << _tagNum << std::endl;
    //   return false;
    // }
    // if (_last_tagNum > 0) {
    //     if ((_last_tag_outputSize != _tag_outputSize) | (_last_tagSize != _tagSize)|(_last_tagDim != _tagDim))
    //       std::cout << "Tag mismatch, end calculation! " << std::endl;
    //       return false;
    // }  
    return true;
  }


  void computerHiddenScore(const vector<Feature>& features, vector<Tensor<xpu, 2, dtype> >& hidden_vector) {
    // cout << "begin computing hidden score " << endl;
    int seq_size = features.size();
    int offset = 0;
    // assign(hidden_vector, 0.0);

    vector<Tensor<xpu, 2, dtype> > input(seq_size);
    vector<Tensor<xpu, 2, dtype> > lstmoutput(seq_size);

    vector<Tensor<xpu, 2, dtype> > i_project_left(seq_size), o_project_left(seq_size), f_project_left(seq_size);
    vector<Tensor<xpu, 2, dtype> > mc_project_left(seq_size), c_project_left(seq_size), my_project_left(seq_size);
    vector<Tensor<xpu, 2, dtype> > i_project_right(seq_size), o_project_right(seq_size), f_project_right(seq_size);
    vector<Tensor<xpu, 2, dtype> > mc_project_right(seq_size), c_project_right(seq_size), my_project_right(seq_size);
    vector<Tensor<xpu, 2, dtype> > project_left(seq_size), project_right(seq_size);

    vector<Tensor<xpu, 2, dtype> > project(seq_size);
    vector<Tensor<xpu, 2, dtype> > output(seq_size);

    vector<Tensor<xpu, 3, dtype> > charprime(seq_size);
    vector<Tensor<xpu, 3, dtype> > charinput(seq_size);
    vector<Tensor<xpu, 3, dtype> > charhidden(seq_size);
    vector<Tensor<xpu, 3, dtype> > chargatedpoolIndex(seq_size);
    vector<Tensor<xpu, 2, dtype> > chargatedpool(seq_size);
    vector<Tensor<xpu, 3, dtype> > chargateweightMiddle(seq_size);
    vector<Tensor<xpu, 3, dtype> > chargateweight(seq_size);
    vector<Tensor<xpu, 2, dtype> > chargateweightsum(seq_size);
    vector<Tensor<xpu, 3, dtype> > tagprime(seq_size);
    vector<Tensor<xpu, 2, dtype> > tagoutput(seq_size);
    vector<Tensor<xpu, 2, dtype> > wordprime(seq_size);
    vector<Tensor<xpu, 2, dtype> > wordrepresent(seq_size);

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = features[idx];

      int char_num = feature.chars.size();
      charprime[idx] = NewTensor<xpu>(Shape3(char_num, 1, _last_charDim), d_zero);
      charinput[idx] = NewTensor<xpu>(Shape3(char_num, 1, _last_char_inputSize), d_zero);
      charhidden[idx] = NewTensor<xpu>(Shape3(char_num, 1, _last_char_outputSize), d_zero);
      chargatedpoolIndex[idx] = NewTensor<xpu>(Shape3(char_num, 1, _last_char_outputSize), d_zero);
      chargatedpool[idx] = NewTensor<xpu>(Shape2(1, _last_char_outputSize), d_zero);
      chargateweightMiddle[idx] = NewTensor<xpu>(Shape3(char_num, 1, _last_char_outputSize), d_zero);
      chargateweight[idx] = NewTensor<xpu>(Shape3(char_num, 1, _last_char_outputSize), d_zero);
      chargateweightsum[idx] = NewTensor<xpu>(Shape2(1, _last_char_outputSize), d_zero);
      if (_last_tagNum > 0) {
        tagprime[idx] = NewTensor<xpu>(Shape3(_last_tagNum, 1, _last_tagDim[0]), d_zero);
        tagoutput[idx] = NewTensor<xpu>(Shape2(1, _last_tag_outputSize), d_zero);
      }
      wordprime[idx] = NewTensor<xpu>(Shape2(1, _last_wordDim), d_zero);
      wordrepresent[idx] = NewTensor<xpu>(Shape2(1, _last_token_representation_size), d_zero);
      input[idx] = NewTensor<xpu>(Shape2(1, _last_inputsize), d_zero);

      i_project_left[idx] = NewTensor<xpu>(Shape2(1, _last_lstmhiddensize), d_zero);
      o_project_left[idx] = NewTensor<xpu>(Shape2(1, _last_lstmhiddensize), d_zero);
      f_project_left[idx] = NewTensor<xpu>(Shape2(1, _last_lstmhiddensize), d_zero);
      mc_project_left[idx] = NewTensor<xpu>(Shape2(1, _last_lstmhiddensize), d_zero);
      c_project_left[idx] = NewTensor<xpu>(Shape2(1, _last_lstmhiddensize), d_zero);
      my_project_left[idx] = NewTensor<xpu>(Shape2(1, _last_lstmhiddensize), d_zero);
      project_left[idx] = NewTensor<xpu>(Shape2(1, _last_lstmhiddensize), d_zero);

      i_project_right[idx] = NewTensor<xpu>(Shape2(1, _last_lstmhiddensize), d_zero);
      o_project_right[idx] = NewTensor<xpu>(Shape2(1, _last_lstmhiddensize), d_zero);
      f_project_right[idx] = NewTensor<xpu>(Shape2(1, _last_lstmhiddensize), d_zero);
      mc_project_right[idx] = NewTensor<xpu>(Shape2(1, _last_lstmhiddensize), d_zero);
      c_project_right[idx] = NewTensor<xpu>(Shape2(1, _last_lstmhiddensize), d_zero);
      my_project_right[idx] = NewTensor<xpu>(Shape2(1, _last_lstmhiddensize), d_zero);
      project_right[idx] = NewTensor<xpu>(Shape2(1, _last_lstmhiddensize), d_zero);

      lstmoutput[idx] = NewTensor<xpu>(Shape2(1, 2 * _last_lstmhiddensize), d_zero);
      project[idx] = NewTensor<xpu>(Shape2(1, _last_hiddensize), d_zero);
      // output[idx] = NewTensor<xpu>(Shape2(1, _last_labelSize), d_zero);
    }

    //forward propagation
    //input setting, and linear setting
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = features[idx];
      //linear features should not be dropped out

      const vector<int>& words = feature.words;
      _last_words.GetEmb(words[0], wordprime[idx]);

      const vector<int>& chars = feature.chars;
      int char_num = chars.size();

      //charprime
      for (int idy = 0; idy < char_num; idy++) {
        _last_chars.GetEmb(chars[idy], charprime[idx][idy]);
      }

      // char context
      windowlized(charprime[idx], charinput[idx], _charcontext);

      // char combination
      _last_tanhchar_project.ComputeForwardScore(charinput[idx], charhidden[idx]);

      // char gated pooling
      _last_gatedchar_pooling.ComputeForwardScore(charhidden[idx], wordprime[idx], chargateweightMiddle[idx], chargateweight[idx], chargateweightsum[idx], chargatedpoolIndex[idx], chargatedpool[idx]);
      // tag prime get
      if (_last_tagNum > 0) {
        const vector<int>& tags = feature.tags;
        for (int idy = 0; idy < _tagNum; idy++){
          _last_tags[idy].GetEmb(tags[idy], tagprime[idx][idy]);
        }
        concat(tagprime[idx], tagoutput[idx]); 
      } 
    }
    if (_last_tagNum > 0) {
      for (int idx = 0; idx < seq_size; idx++) {
        concat(wordprime[idx], chargatedpool[idx], tagoutput[idx], wordrepresent[idx]);
      }
    }
    else {
      for (int idx = 0; idx < seq_size; idx++) {
        concat(wordprime[idx], chargatedpool[idx], wordrepresent[idx]);
      }      
    }

    windowlized(wordrepresent, input, _wordcontext);

    _lastrnn_left_project.ComputeForwardScore(input, i_project_left, o_project_left, f_project_left, mc_project_left,
        c_project_left, my_project_left, project_left);
    _lastrnn_right_project.ComputeForwardScore(input, i_project_right, o_project_right, f_project_right, mc_project_right,
        c_project_right, my_project_right, project_right);

    for (int idx = 0; idx < seq_size; idx++) {
      concat(project_left[idx], project_right[idx], lstmoutput[idx]);
    }
    _last_tanh_project.ComputeForwardScore(lstmoutput, project);
    // std::cout << "line 964, before equal project " << std::endl;
    for (int idx = 0; idx < seq_size; idx++) {
      for (int idy = 0; idy < _last_hiddensize; idy++) {
        hidden_vector[idx][0][idy] = project[idx][0][idy];
        // cout << "position " << idx << ", " << idy << ", " << seq_size << ", " << _last_hiddensize << " hidden = "<< hidden_vector[idx][0][idy] << "   , project = " << project[idx][0][idy] <<endl;
      }
    }
    // std::cout << hidden_vector[0][1] << "with "<< project[0][1] << std::endl;

    // _last_olayer_linear.ComputeForwardScore(project, output);

    // decode algorithm
    // _crf_layer.predict(output, results);

    //release

    for (int idx = 0; idx < seq_size; idx++) {
      FreeSpace(&(charprime[idx]));
      FreeSpace(&(charinput[idx]));
      FreeSpace(&(charhidden[idx]));
      FreeSpace(&(chargatedpoolIndex[idx]));
      FreeSpace(&(chargatedpool[idx]));
      FreeSpace(&(chargateweightMiddle[idx]));
      FreeSpace(&(chargateweight[idx]));
      FreeSpace(&(chargateweightsum[idx]));
      if (_last_tagNum > 0) {
        FreeSpace(&(tagprime[idx]));
        FreeSpace(&(tagoutput[idx]));
      }
      FreeSpace(&(wordprime[idx]));
      FreeSpace(&(wordrepresent[idx]));
      FreeSpace(&(input[idx]));

      FreeSpace(&(i_project_left[idx]));
      FreeSpace(&(o_project_left[idx]));
      FreeSpace(&(f_project_left[idx]));
      FreeSpace(&(mc_project_left[idx]));
      FreeSpace(&(c_project_left[idx]));
      FreeSpace(&(my_project_left[idx]));
      FreeSpace(&(project_left[idx]));

      FreeSpace(&(i_project_right[idx]));
      FreeSpace(&(o_project_right[idx]));
      FreeSpace(&(f_project_right[idx]));
      FreeSpace(&(mc_project_right[idx]));
      FreeSpace(&(c_project_right[idx]));
      FreeSpace(&(my_project_right[idx]));
      FreeSpace(&(project_right[idx]));

      FreeSpace(&(lstmoutput[idx]));
      FreeSpace(&(project[idx]));
      // FreeSpace(&(output[idx]));

    }
    // cout << "end computer hidden score" << endl;
  }



  void writeModel();

  void loadModel();



  

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter) {
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols;
    idRows.clear();
    idCols.clear();
    for (int i = 0; i < Wd.size(0); ++i)
      idRows.push_back(i);
    for (int idx = 0; idx < Wd.size(1); idx++)
      idCols.push_back(idx);

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());

    int check_i = idRows[0], check_j = idCols[0];

    dtype orginValue = Wd[check_i][check_j];

    Wd[check_i][check_j] = orginValue + 0.1;
    dtype lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossAdd += computeScore(oneExam);
    }

    Wd[check_i][check_j] = orginValue - 0.1;
    dtype lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossPlus += computeScore(oneExam);
    }

    dtype mockGrad = (lossAdd - lossPlus) / 0.2;
    mockGrad = mockGrad / examples.size();
    dtype computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;
  }

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter,
      const hash_set<int>& indexes, bool bRow = true) {
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols;
    idRows.clear();
    idCols.clear();
    static hash_set<int>::iterator it;
    if (bRow) {
      for (it = indexes.begin(); it != indexes.end(); ++it)
        idRows.push_back(*it);
      for (int idx = 0; idx < Wd.size(1); idx++)
        idCols.push_back(idx);
    } else {
      for (it = indexes.begin(); it != indexes.end(); ++it)
        idCols.push_back(*it);
      for (int idx = 0; idx < Wd.size(0); idx++)
        idRows.push_back(idx);
    }

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());

    int check_i = idRows[0], check_j = idCols[0];

    dtype orginValue = Wd[check_i][check_j];

    Wd[check_i][check_j] = orginValue + 0.1;
    dtype lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossAdd += computeScore(oneExam);
    }

    Wd[check_i][check_j] = orginValue - 0.1;
    dtype lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossPlus += computeScore(oneExam);
    }

    dtype mockGrad = (lossAdd - lossPlus) / 0.2;
    mockGrad = mockGrad / examples.size();
    dtype computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;

  }

  void checkgrads(const vector<Example>& examples, int iter) {
    checkgrad(examples, _olayer_linear._W, _olayer_linear._gradW, "_olayer_linear._W", iter);
    checkgrad(examples, _tanh_project._W, _tanh_project._gradW, "_tanh_project._W", iter);

    checkgrad(examples, _tanhchar_project._W, _tanhchar_project._gradW, "_tanhchar_project._W", iter);
    checkgrad(examples, _tanhchar_project._b, _tanhchar_project._gradb, "_tanhchar_project._b", iter);

    checkgrad(examples, _gatedchar_pooling._bi_gates._WL, _gatedchar_pooling._bi_gates._gradWL, "_gatedchar_pooling._bi_gates._WL", iter);
    checkgrad(examples, _gatedchar_pooling._bi_gates._WR, _gatedchar_pooling._bi_gates._gradWR, "_gatedchar_pooling._bi_gates._WR", iter);
    checkgrad(examples, _gatedchar_pooling._bi_gates._b, _gatedchar_pooling._bi_gates._gradb, "_gatedchar_pooling._bi_gates._b", iter);

    checkgrad(examples, _gatedchar_pooling._uni_gates._W, _gatedchar_pooling._uni_gates._gradW, "_gatedchar_pooling._uni_gates._W", iter);
    checkgrad(examples, _gatedchar_pooling._uni_gates._b, _gatedchar_pooling._uni_gates._gradb, "_gatedchar_pooling._uni_gates._b", iter);

    checkgrad(examples, rnn_left_project._lstm_output._W1, rnn_left_project._lstm_output._gradW1, "rnn_left_project._lstm_output._W1", iter);
    checkgrad(examples, rnn_left_project._lstm_output._W2, rnn_left_project._lstm_output._gradW2, "rnn_left_project._lstm_output._W2", iter);
    checkgrad(examples, rnn_left_project._lstm_output._W3, rnn_left_project._lstm_output._gradW3, "rnn_left_project._lstm_output._W3", iter);
    checkgrad(examples, rnn_left_project._lstm_output._b, rnn_left_project._lstm_output._gradb, "rnn_left_project._lstm_output._b", iter);
    checkgrad(examples, rnn_left_project._lstm_input._W1, rnn_left_project._lstm_input._gradW1, "rnn_left_project._lstm_input._W1", iter);
    checkgrad(examples, rnn_left_project._lstm_input._W2, rnn_left_project._lstm_input._gradW2, "rnn_left_project._lstm_input._W2", iter);
    checkgrad(examples, rnn_left_project._lstm_input._W3, rnn_left_project._lstm_input._gradW3, "rnn_left_project._lstm_input._W3", iter);
    checkgrad(examples, rnn_left_project._lstm_input._b, rnn_left_project._lstm_input._gradb, "rnn_left_project._lstm_input._b", iter);
    checkgrad(examples, rnn_left_project._lstm_forget._W1, rnn_left_project._lstm_forget._gradW1, "rnn_left_project._lstm_forget._W1", iter);
    checkgrad(examples, rnn_left_project._lstm_forget._W2, rnn_left_project._lstm_forget._gradW2, "rnn_left_project._lstm_forget._W2", iter);
    checkgrad(examples, rnn_left_project._lstm_forget._W3, rnn_left_project._lstm_forget._gradW3, "rnn_left_project._lstm_forget._W3", iter);
    checkgrad(examples, rnn_left_project._lstm_forget._b, rnn_left_project._lstm_forget._gradb, "rnn_left_project._lstm_forget._b", iter);
    checkgrad(examples, rnn_left_project._lstm_cell._WL, rnn_left_project._lstm_cell._gradWL, "rnn_left_project._lstm_cell._WL", iter);
    checkgrad(examples, rnn_left_project._lstm_cell._WR, rnn_left_project._lstm_cell._gradWR, "rnn_left_project._lstm_cell._WR", iter);
    checkgrad(examples, rnn_left_project._lstm_cell._b, rnn_left_project._lstm_cell._gradb, "rnn_left_project._lstm_cell._b", iter);


    checkgrad(examples, rnn_right_project._lstm_output._W1, rnn_right_project._lstm_output._gradW1, "rnn_right_project._lstm_output._W1", iter);
    checkgrad(examples, rnn_right_project._lstm_output._W2, rnn_right_project._lstm_output._gradW2, "rnn_right_project._lstm_output._W2", iter);
    checkgrad(examples, rnn_right_project._lstm_output._W3, rnn_right_project._lstm_output._gradW3, "rnn_right_project._lstm_output._W3", iter);
    checkgrad(examples, rnn_right_project._lstm_output._b, rnn_right_project._lstm_output._gradb, "rnn_right_project._lstm_output._b", iter);
    checkgrad(examples, rnn_right_project._lstm_input._W1, rnn_right_project._lstm_input._gradW1, "rnn_right_project._lstm_input._W1", iter);
    checkgrad(examples, rnn_right_project._lstm_input._W2, rnn_right_project._lstm_input._gradW2, "rnn_right_project._lstm_input._W2", iter);
    checkgrad(examples, rnn_right_project._lstm_input._W3, rnn_right_project._lstm_input._gradW3, "rnn_right_project._lstm_input._W3", iter);
    checkgrad(examples, rnn_right_project._lstm_input._b, rnn_right_project._lstm_input._gradb, "rnn_right_project._lstm_input._b", iter);
    checkgrad(examples, rnn_right_project._lstm_forget._W1, rnn_right_project._lstm_forget._gradW1, "rnn_right_project._lstm_forget._W1", iter);
    checkgrad(examples, rnn_right_project._lstm_forget._W2, rnn_right_project._lstm_forget._gradW2, "rnn_right_project._lstm_forget._W2", iter);
    checkgrad(examples, rnn_right_project._lstm_forget._W3, rnn_right_project._lstm_forget._gradW3, "rnn_right_project._lstm_forget._W3", iter);
    checkgrad(examples, rnn_right_project._lstm_forget._b, rnn_right_project._lstm_forget._gradb, "rnn_right_project._lstm_forget._b", iter);
    checkgrad(examples, rnn_right_project._lstm_cell._WL, rnn_right_project._lstm_cell._gradWL, "rnn_right_project._lstm_cell._WL", iter);
    checkgrad(examples, rnn_right_project._lstm_cell._WR, rnn_right_project._lstm_cell._gradWR, "rnn_right_project._lstm_cell._WR", iter);
    checkgrad(examples, rnn_right_project._lstm_cell._b, rnn_right_project._lstm_cell._gradb, "rnn_right_project._lstm_cell._b", iter);
    if (_tagNum > 0) {
      checkgrad(examples, _crf_layer._tagBigram, _crf_layer._grad_tagBigram, "_crf_layer._tagBigram", iter);
    }
    checkgrad(examples, _words._E, _words._gradE, "_words._E", iter, _words._indexers);
    checkgrad(examples, _chars._E, _chars._gradE, "_chars._E", iter, _chars._indexers);
    // tag checkgrad
    if (_tagNum > 0) {
      for (int i = 0; i < _tagNum; i++){
        checkgrad(examples, _tags[i]._E, _tags[i]._gradE, "_tags._E", iter, _tags[i]._indexers);
      }
    }

  }

  void copyParameters() {
  _last_words = _words;
  _last_chars = _chars;
  // tag variables
  _last_tagNum = _tagNum;
  if (_last_tagNum > 0) {
    _last_tag_outputSize = _tag_outputSize;
    _last_tagSize = _tagSize;
    _last_tagDim = _tagDim;
    _last_tags = _tags;
  }

  _last_wordcontext = _wordcontext;
  _last_wordwindow = _wordwindow;
  _last_wordSize = _wordSize;
  _last_wordDim = _wordDim;

  _last_charcontext = _charcontext;
  _last_charwindow = _charwindow;
  _last_charSize = _charSize;
  _last_charDim = _charDim;
  _last_char_outputSize = _char_outputSize;
  _last_char_inputSize = _char_inputSize;

  _last_lstmhiddensize = _lstmhiddensize;
  _last_hiddensize = _hiddensize;
  _last_inputsize = _inputsize;
  _last_token_representation_size = _token_representation_size;
  _last_olayer_linear = _olayer_linear;
  _last_tanh_project = _tanh_project;
  _last_tanhchar_project = _tanhchar_project;
  _last_gatedchar_pooling = _gatedchar_pooling;
  _lastrnn_left_project = rnn_left_project;
  _lastrnn_right_project = rnn_right_project;
  _last_crf_layer = _crf_layer;

  _last_labelSize = _labelSize;

  _last_eval = _eval;

  _last_dropOut = _dropOut;
  

  }

public:
  inline void resetEval() {
    _eval.reset();
  }

  inline void setDropValue(dtype dropOut) {
    _dropOut = dropOut;
  }

  inline void setWordEmbFinetune(bool b_wordEmb_finetune) {
    _words.setEmbFineTune(b_wordEmb_finetune);
  }
  
  inline void setTagEmbFinetune(bool b_tagEmb_finetune) {
    if (_tagNum > 0) {
      for (int idx = 0; idx < _tagNum; idx++){
        _tags[idx].setEmbFineTune(b_tagEmb_finetune);
      }  
    } 
  }


};

#endif /* SRC_LSTMCRFMLClassifier_H_ */
