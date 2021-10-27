//
// Created by perseusdg on 10/21/21.
//

#ifndef TKDNN_REFLECTIONPADDING2DRT_H
#define TKDNN_REFLECTIONPADDING2DRT_H

#include<cassert>
#include "./kernels.h"

class ReflectionPadding2DRT : public IPlugin {
public :
    ReflectionPadding2DRT(int64_t pad) {
        this->pad = pad;
        for (int i = 0; i < 4; i++) {
            this->pad_lrtb[i] = pad;
        }
        this->batch = 1;
    }

    ~ReflectionPadding2DRT() {
    }

    int getNbOutputs() const override{
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override{
        return DimsCHW{ inputs[0].d[0],inputs[0].d[1]+2*static_cast<int32_t>(pad),inputs[0].d[2]+2*static_cast<int32_t>(pad)};
    }

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override{
        c = inputDims[0].d[0];
        input_h = inputDims[0].d[1];
        input_w = inputDims[0].d[2];
    }

    int initialize() override {
        return 0;
    }

    virtual void terminate() override {
    }

    virtual size_t getWorkspaceSize(int maxBatchSize) const override {
        return 0;
    }

    virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override {
        dnnType* srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
        dnnType* dstData = reinterpret_cast<dnnType*>(outputs[0]);
        reflection_pad2d_out_forward(pad_lrtb, srcData, dstData, input_h, input_w, c, batch, stream);
        return 0;
    }

    virtual size_t getSerializationSize() override{
        return (8 * sizeof(int64_t));

    }

    virtual void serialize(void* buffer) override{
        char* buf = reinterpret_cast<char*>(buffer), * a = buf;
        tk::dnn::writeBUF(buf, pad);
        for (int i = 0; i < 4; i++) {
            tk::dnn::writeBUF(buf, pad_lrtb[i]);
        }
        tk::dnn::writeBUF(buf, input_h);
        tk::dnn::writeBUF(buf, input_w);
        tk::dnn::writeBUF(buf, batch);
        tk::dnn::writeBUF(buf, c);
        assert(buf == a + getSerializationSize());

    }
    
    int64_t pad_lrtb[4], input_h, input_w,batch,c;

protected:
    int64_t pad; 
 

};


#endif //TKDNN_REFLECTIONPADDING2DRT_H
