cd $(dirname $0)
cd lib
python setup.py build_ext --inplace
cd roi_pooling_layer
./make.sh
cd ../top
make
