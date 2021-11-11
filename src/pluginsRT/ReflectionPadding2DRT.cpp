#include <tkDNN/pluginsRT/ReflectionPadding2DRT.h>
using namespace nvinfer1;

std::vector<PluginField> ReflectionPadding2DRTPluginCreator::mPluginAttributes;
PluginFieldCollection ReflectionPadding2DRTPluginCreator::mFC{};


static const char* REFLECTIONPADDING2DRT_PLUGIN_VERSION{"1"};
static const char* REFLECTIONPADDING2DRT_PLUGIN_NAME{"ReflectionPadding2D_tkDNN"};


ReflectionPadding2DRT::ReflectionPadding2DRT(int32_t pad,int32_t inputH,int32_t inputW,int32_t batch,int32_t c) {
    this->padding = pad;
    for (int i = 0; i < 4; i++) {
        this->pad_lrtb[i] = pad;
    }
    this->input_h = inputH;
    this->input_w = inputW;
    this->batch = batch;
    this->c = c;
}

ReflectionPadding2DRT::ReflectionPadding2DRT(const void *data, size_t length) {
    const char* buf = reinterpret_cast<const char*>(data),*bufCheck=buf;
    padding = readBUF<int32_t>(buf);
    input_h = readBUF<int32_t>(buf);
    input_w = readBUF<int32_t>(buf);
    batch = readBUF<int32_t>(buf);
    c = readBUF<int32_t>(buf);
}

ReflectionPadding2DRT::~ReflectionPadding2DRT() {

}

int ReflectionPadding2DRT::getNbOutputs() const NOEXCEPT {
    return 1;
}

Dims ReflectionPadding2DRT::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT {
    return Dims3{ inputs[0].d[0],inputs[0].d[1]+2*static_cast<int32_t>(padding),inputs[0].d[2]+2*static_cast<int32_t>(padding)};
}

int ReflectionPadding2DRT::initialize() NOEXCEPT {
    return 0;
}

void ReflectionPadding2DRT::terminate() NOEXCEPT {

}

size_t ReflectionPadding2DRT::getWorkspaceSize(int maxBatchSize) const NOEXCEPT {
    return 0;
}

#if NV_TENSORRT_MAJOR > 7
int ReflectionPadding2DRT::enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                                   cudaStream_t stream) NOEXCEPT {
    dnnType* srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
    dnnType* dstData = reinterpret_cast<dnnType*>(outputs[0]);
    reflection_pad2d_out_forward(pad_lrtb, srcData, dstData, input_h, input_w, c, batch, stream);
    return 0;
}


void ReflectionPadding2DRT::destroy() NOEXCEPT {
    delete this;
}

void ReflectionPadding2DRT::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

size_t ReflectionPadding2DRT::getSerializationSize() const NOEXCEPT {
    return 5*sizeof(int32_t);
}

void ReflectionPadding2DRT::serialize(void *buffer) const NOEXCEPT {
    char *buf = reinterpret_cast<char*>(buffer),*a=buf;
    writeBUF(buf,padding);
    writeBUF(buf,input_h);
    writeBUF(buf,input_w);
    writeBUF(buf,batch);
    writeBUF(buf,c);
}

#elif NV_TENSORRT_MAJOR <= 7
int32_t ReflectionPadding2DRT::enqueue (int32_t batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) {
    dnnType* srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
    dnnType* dstData = reinterpret_cast<dnnType*>(outputs[0]);
    reflection_pad2d_out_forward(pad_lrtb, srcData, dstData, input_h, input_w, c, batch, stream);
    return 0;
}
#endif

bool ReflectionPadding2DRT::supportsFormat(DataType type, PluginFormat format) const NOEXCEPT {
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

const char *ReflectionPadding2DRT::getPluginType() const NOEXCEPT {
    return REFLECTIONPADDING2DRT_PLUGIN_NAME;
}

const char *ReflectionPadding2DRT::getPluginVersion() const NOEXCEPT {
    return REFLECTIONPADDING2DRT_PLUGIN_VERSION;
}

const char *ReflectionPadding2DRT::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

bool ReflectionPadding2DRT::isOutputBroadcastAcrossBatch(int32_t outputIndex, const bool *inputIsBroadcasted,
                                                         int32_t nbInputs) const NOEXCEPT {
    return false;
}

bool ReflectionPadding2DRT::canBroadcastInputAcrossBatch(int32_t inputIndex) const NOEXCEPT {
    return false;
}

void ReflectionPadding2DRT::configurePlugin(const Dims *inputDims, int32_t nbInputs, const Dims *outputDims,
                                            int32_t nbOutputs, const DataType *inputTypes, const DataType *outputTypes,
                                            const bool *inputIsBroadcast, const bool *outputIsBroadcast,
                                            PluginFormat floatFormat, int32_t maxBatchSize) NOEXCEPT {

}

void ReflectionPadding2DRT::attachToContext(cudnnContext *, cublasContext *, IGpuAllocator *) NOEXCEPT {

}

void ReflectionPadding2DRT::detachFromContext() NOEXCEPT {

}

DataType ReflectionPadding2DRT::getOutputDataType(int32_t index, const nvinfer1::DataType *inputTypes,
                                                  int32_t nbInputs) const NOEXCEPT {
    return DataType::kFLOAT;
}

IPluginV2Ext *ReflectionPadding2DRT::clone() const NOEXCEPT {
    auto* p = new ReflectionPadding2DRT(padding,input_h,input_w,batch,c);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

ReflectionPadding2DRTPluginCreator::ReflectionPadding2DRTPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void ReflectionPadding2DRTPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *ReflectionPadding2DRTPluginCreator::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

IPluginV2Ext *ReflectionPadding2DRTPluginCreator::deserializePlugin(const char *name, const void *serialData,
                                                                    size_t serialLength) NOEXCEPT {
    auto *pluginObj = new ReflectionPadding2DRT(serialData,serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

IPluginV2Ext *ReflectionPadding2DRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT {
    const PluginField *fields = fc->fields;
    int padding = *(static_cast<const int32_t*>(fields[0].data));
    int inputH = *(static_cast<const int32_t*>(fields[1].data));
    int inputW = *(static_cast<const int32_t*>(fields[2].data));
    int batch = *(static_cast<const int32_t*>(fields[3].data));
    int c = *(static_cast<const int32_t*>(fields[4].data));
    auto *pluginObj = new ReflectionPadding2DRT(padding,inputH,inputW,batch,c);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char *ReflectionPadding2DRTPluginCreator::getPluginName() const NOEXCEPT {
    return REFLECTIONPADDING2DRT_PLUGIN_NAME;
}

const char *ReflectionPadding2DRTPluginCreator::getPluginVersion() const NOEXCEPT {
    return REFLECTIONPADDING2DRT_PLUGIN_VERSION;
}

const PluginFieldCollection *ReflectionPadding2DRTPluginCreator::getFieldNames() NOEXCEPT {
    return &mFC;
}





