.PHONY: clean links

clean :
	-$(RM) -rf *.pyc __pycache__ *.so \
	cython_sampler.cpp \
	cython_utils.cpp \
	norm_aggr.cpp \
	tf/__pycache__ \
	tf/*.pyc

links :
	ln -sf cython_sampler.cpython-37m-x86_64-linux-gnu.so cython_sampler.so
	ln -sf cython_utils.cpython-37m-x86_64-linux-gnu.so cython_utils.so
	ln -sf norm_aggr.cpython-37m-x86_64-linux-gnu.so norm_aggr.so
