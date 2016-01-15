 make CDOLSTMLabeler
 wait
 ./CDOLSTMLabeler -p -train example/pos.train -dev example/pos.dev -option example/pos.option -word example/sena.emb -lastmodel example/pos.lastmodel
 wait
 ./CDOLSTMLabeler -l -train example/Entity.train -dev example/Entity.dev -option example/demo.option -word example/sena.emb -lastmodel example/pos.lastmodel -model example/Entity.model
 wait
 ./CDOLSTMLabeler -test example/Entity.test -model example/Entity.model -output example/Entity.test.output
#make SparseGatedCRFMLLabeler
# wait
# ./SparseLSTMCRFMLLabeler -p -train example/pos.train -dev example/pos.dev -option example/pos.option -word example/sena.emb -lastmodel example/pos.lastmodel
#wait
#./SparseGatedCRFMLLabeler -l -train example/Entity.train -dev example/Entity.dev -test example/Entity.test -option example/demo.option -word example/sena.emb -model example/Entity.model
#wait
#./SparseGatedCRFMLLabeler -test example/Entity.test -model example/Entity.model -output example/Entity.test.output
