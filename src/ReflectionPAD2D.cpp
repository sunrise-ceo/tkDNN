#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tk {namespace dnn {

	ReflectionPAD2D::ReflectionPAD2D(Network* net,int64_t pad) : Layer(net) {
		this->pad = pad;
		output_dim.n = input_dim.n;
		output_dim.c = input_dim.c;
		output_dim.h = input_dim.h + 2 * (this->pad);
		output_dim.w = input_dim.w + 2 * (this->pad);
		checkCuda(cudaMalloc(&dstData, output_dim.tot() * sizeof(dnnType)));
	}

	ReflectionPAD2D::~ReflectionPAD2D() {
		checkCuda(cudaFree(dstData));
	}

	dnnType* ReflectionPAD2D::infer(dataDim_t& dim, dnnType* srcData)
	{
		int64_t pad_lrtb[4];
		for (int i = 0; i < 4; i++) {
			pad_lrtb[i] = pad;
		}
		fill(dstData, output_dim.tot(), 0.0);
		reflection_pad2d_out_forward(pad_lrtb,srcData,dstData,input_dim.h,input_dim.w,input_dim.c,1);
		dim = output_dim;
		return dstData;
	}

	
}}