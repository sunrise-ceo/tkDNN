#include <tkDNN/pluginsRT/FlattenConcatRT.h>
namespace  nvinfer1 {

    std::vector<PluginField> FlattenConcatRTPluginCreator::mPluginAttributes;
    PluginFieldCollection FlattenConcatRTPluginCreator::mFC{};

    FlattenConcatRT::FlattenConcatRT() {
        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("CUBLAS initialization failed\n");
            return;
        }
    }

    FlattenConcatRT::FlattenConcatRT(const void *data, size_t length) {
        const char *buf = reinterpret_cast<const char *>(data), *bufCheck = buf;
        c = readBUF<int>(buf);
        h = readBUF<int>(buf);
        w = readBUF<int>(buf);
        rows = readBUF<int>(buf);
        cols = readBUF<int>(buf);
        assert(buf == bufCheck + length);
    }

    FlattenConcatRT::~FlattenConcatRT() {}

    int FlattenConcatRT::getNbOutputs() const NOEXCEPT {
        return 1;
    }

    Dims FlattenConcatRT::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT {
        return Dims3{inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2], 1, 1};
    }

    void
    FlattenConcatRT::configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,
                                         DataType type, PluginFormat format, int maxBatchSize) NOEXCEPT {
        assert(nbOutputs == 1 && nbInputs == 1);
        rows = inputDims[0].d[0];
        cols = inputDims[0].d[1] * inputDims[0].d[2];
        c = inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2];
        h = 1;
        w = 1;
    }

    int FlattenConcatRT::initialize() NOEXCEPT {
        return 0;
    }

    void FlattenConcatRT::terminate() NOEXCEPT {
        checkERROR(cublasDestroy(handle));
    }

    size_t FlattenConcatRT::getWorkspaceSize(int maxBatchSize) const NOEXCEPT {
        return 0;
    }

    int FlattenConcatRT::enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                                 cudaStream_t stream) NOEXCEPT {
        dnnType *srcData = (dnnType *) reinterpret_cast<const dnnType *>(inputs[0]);
        dnnType *dstData = reinterpret_cast<dnnType *>(outputs[0]);
        checkCuda(cudaMemcpyAsync(dstData, srcData, batchSize * rows * cols * sizeof(dnnType), cudaMemcpyDeviceToDevice,
                                  stream));

        checkERROR(cublasSetStream(handle, stream));
        for (int i = 0; i < batchSize; i++) {
            float const alpha(1.0);
            float const beta(0.0);
            int offset = i * rows * cols;
            checkERROR(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, &alpha, srcData + offset, cols, &beta,
                                   srcData + offset, rows, dstData + offset, rows));
        }
        return 0;
    }

    size_t FlattenConcatRT::getSerializationSize() const NOEXCEPT {
        return 5 * sizeof(int);
    }

    void FlattenConcatRT::serialize(void *buffer) const NOEXCEPT {
        char *buf = reinterpret_cast<char *>(buffer), *a = buf;
        writeBUF(buf, c);
        writeBUF(buf, h);
        writeBUF(buf, w);
        writeBUF(buf, rows);
        writeBUF(buf, cols);
        assert(buf == a + getSerializationSize());
    }

    void FlattenConcatRT::destroy() NOEXCEPT {
        delete this;
    }

    bool FlattenConcatRT::supportsFormat(DataType type, PluginFormat format) const NOEXCEPT {
        return true;
    }

    const char *FlattenConcatRT::getPluginType() const NOEXCEPT {
        return "FlattenConcatRT_tkDNN";
    }

    const char *FlattenConcatRT::getPluginVersion() const NOEXCEPT {
        return "1";
    }

    const char *FlattenConcatRT::getPluginNamespace() const NOEXCEPT {
        return mPluginNamespace.c_str();
    }

    void FlattenConcatRT::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
        mPluginNamespace = pluginNamespace;
    }

    IPluginV2 *FlattenConcatRT::clone() const NOEXCEPT {
        auto *p = new FlattenConcatRT();
        p->setPluginNamespace(mPluginNamespace.c_str());
        return p;
    }


    FlattenConcatRTPluginCreator::FlattenConcatRTPluginCreator() {
        mPluginAttributes.clear();
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    void FlattenConcatRTPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
        mPluginNamespace = pluginNamespace;
    }

    const char *FlattenConcatRTPluginCreator::getPluginNamespace() const NOEXCEPT {
        return mPluginNamespace.c_str();
    }

    IPluginV2 *FlattenConcatRTPluginCreator::deserializePlugin(const char *name, const void *serialData,
                                                               size_t serialLength) NOEXCEPT {
        auto *pluginObj = new FlattenConcatRT(serialData, serialLength);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    IPluginV2 *FlattenConcatRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT {
        auto *pluginObj = new FlattenConcatRT();
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    const char *FlattenConcatRTPluginCreator::getPluginName() const NOEXCEPT {
        return "FlattenConcatRT_tkDNN";
    }

    const char *FlattenConcatRTPluginCreator::getPluginVersion() const NOEXCEPT {
        return "1";
    }

    const PluginFieldCollection *FlattenConcatRTPluginCreator::getFieldNames() NOEXCEPT {
        return &mFC;
    }
}






