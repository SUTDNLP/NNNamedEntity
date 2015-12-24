/*
 * Labeler.h
 *
 *  Created on: Mar 16, 2015
 *      Author: mszhang
 * Use double layer LSTMCRFML, pretrain first LSTM, train second LSTM using first LSTM hidden layer output as word emb
 */

#ifndef SRC_DLSTMCRFMLLABELER_H_
#define SRC_DLSTMCRFMLLABELER_H_


#include "N3L.h"
#include "basic/DLSTMCRFMLClassifier.h"
#include "Options.h"
#include "Instance.h"
#include "Example.h"


#include "Pipe.h"
#include "Utf.h"

using namespace nr;
using namespace std;

class Labeler {

public:
  std::string nullkey;
  std::string unknownkey;
  std::string seperateKey;

public:
  Alphabet m_featAlphabet;
  Alphabet m_labelAlphabet;

  Alphabet m_wordAlphabet;
  Alphabet m_charAlphabet;

  NRVec<Alphabet> m_tagAlphabets;

  Alphabet last_featAlphabet;
  Alphabet last_labelAlphabet;

  Alphabet last_wordAlphabet;
  Alphabet last_charAlphabet;

  NRVec<Alphabet> last_tagAlphabets;

public:
  Options m_options;
  Options last_options;
  bool second_LSTM = false;

  Pipe m_pipe;

#if USE_CUDA==1
  DLSTMCRFMLClassifier<gpu> m_classifier;
  DLSTMCRFMLClassifier<gpu> last_classifier;

#else
  DLSTMCRFMLClassifier<cpu> m_classifier;
  DLSTMCRFMLClassifier<cpu> last_classifier;
#endif

public:
  void readWordEmbeddings(const string& inFile, NRMat<dtype>& wordEmb);

public:
  Labeler();
  virtual ~Labeler();

public:

  int createAlphabet(const vector<Instance>& vecInsts);

  int addTestWordAlpha(const vector<Instance>& vecInsts);
  int addTestCharAlpha(const vector<Instance>& vecInsts);
  int addTestTagAlpha(const vector<Instance>& vecInsts);

  void extractLinearFeatures(vector<string>& features, const Instance* pInstance, int idx);
  void extractFeature(Feature& feat, const Instance* pInstance, int idx);

  void convert2Example(const Instance* pInstance, Example& exam);
  void initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams);

  void last_extractFeature(Feature& feat, const Instance* pInstance, int idx);

  void last_convert2Example(const Instance* pInstance, Example& exam);
  void last_initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams);

public:
  void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile, const string& wordEmbFile, const string& charEmbFile, const string& lastModelFile);
  int fullpredict(const vector<Feature>& features, const vector<Feature>& last_features, vector<string>& outputs, const vector<string>& words);
  int predict(const vector<Feature>& features, vector<string>& outputs, const vector<string>& words);
  void test(const string& testFile, const string& outputFile, const string& modelFile);

  void writeModelFile(const string& outputModelFile);
  void loadModelFile(const string& inputModelFile);
  void writeLastModelFile(const string& outputModelFile);
  void loadLastModelFile(const string& inputModelFile);

  void enableSecondLSTM();
  void disableSecondLSTM();

};

#endif /* SRC_DLSTMCRFMLLABELER_H_ */
