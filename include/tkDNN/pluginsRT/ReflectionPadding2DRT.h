//
// Created by perseusdg on 10/21/21.
//

#ifndef TKDNN_REFLECTIONPADDING2DRT_H
#define TKDNN_REFLECTIONPADDING2DRT_H

#include<cassert>
#include "./kernels.h"

class ReflectionPadding2DRT : public IPlugin {
public :
    ReflectionPadding2DRT(int pad) {
        this->pad = pad;
    }

    ~ReflectionPadding2DRT();

    int getNbOutputs() const override{
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override{

    }

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override{

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

    }

    virtual size_t getSerializationSize() override{

    }

    virtual void serialize(void* buffer) override{

    }


protected:
    int pad;

};


#endif //TKDNN_REFLECTIONPADDING2DRT_H
