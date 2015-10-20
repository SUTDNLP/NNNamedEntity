cmake .
wait
make SparseTNNCRFMLLabeler
wait
./SparseTNNCRFMLLabeler -l -train example/Entity.train -dev example/Entity.dev -test example/Entity.test -option example/demo.option -word example/sena.emb
