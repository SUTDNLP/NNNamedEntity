cmake .
make SparseTNNCRFMLLabeler
./SparseTNNCRFMLLabeler -l -train example/Entity.train -dev example/Entity.dev -option example/demo.option -word example/sena.emb -model example/demo.model
./SparseTNNCRFMLLabeler -test example/Entity.test -model example/demo.model -output example/Entity.test.output