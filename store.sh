make SparseCRFMLLabeler
wait
./SparseCRFMLLabeler -l -train example/Entity.train -dev example/Entity.dev -test example/Entity.test -option example/demo.option -word example/sena.emb -model example/Entity.model
wait
./SparseCRFMLLabeler -test example/Entity.test -model example/Entity.model -output example/Entity.test.output0