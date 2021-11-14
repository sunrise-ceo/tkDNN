#include <tkDNN/pluginsRT/RegionRT.h>
using namespace nvinfer1;

std::vector<PluginField> RegionRTPluginCreator::mPluginAttributes;
PluginFieldCollection RegionRTPluginCreator::mFC{};

static const char* REGIONRT_PLUGIN_VERSION{"1"};
static const char* REGIONRT_PLUGIN_NAME{"RegionRT_tkDNN"};

static const int REGIONRT_CLASSNAME_W = 256;


RegionRT::RegionRT(int classes, int coords, int num, int c, int h, int w, 
                   std::vector<std::string> classNames, std::vector<float> bias_vec) {
    this->classes = classes;
    this->coords = coords;
    this->num = num;
    this->c = c;
    this->h = h;
    this->w = w;
    
    this->classNames = std::move(classNames);
    this->bias = std::move(bias_vec);    
}

RegionRT::~RegionRT() {}

RegionRT::RegionRT(const void *data, size_t length) {
    const char *buf = reinterpret_cast<const char*>(data),*bufCheck=buf;
    classes = readBUF<int>(buf);
    coords = readBUF<int>(buf);
    num = readBUF<int>(buf);
    c = readBUF<int>(buf);
    h = readBUF<int>(buf);
    w = readBUF<int>(buf);
       
    // read anchors' coordinates
    bias.resize(num * 2);
    for (int i = 0; i < num*2; i++) {
        bias[i] = readBUF<dnnType>(buf);
    }
    
    // read class names
    classNames.resize(classes);
    for (int i = 0; i < classes; i++) {
        char tmp[REGIONRT_CLASSNAME_W];
        for (int j = 0; j < REGIONRT_CLASSNAME_W; j++) {
            tmp[j] = readBUF<char>(buf);
        }
        classNames[i] = std::string(tmp);
    }
    
    assert(buf == bufCheck+length);
}

int RegionRT::getNbOutputs() const NOEXCEPT {
    return 1;
}

Dims RegionRT::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT {
    return inputs[0];
}


int RegionRT::initialize() NOEXCEPT {return 0;}

void RegionRT::terminate() NOEXCEPT {}

size_t RegionRT::getWorkspaceSize(int maxBatchSize) const NOEXCEPT { return 0; }

#if NV_TENSORRT_MAJOR > 7
int RegionRT::enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                      cudaStream_t stream) NOEXCEPT {
    dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
    dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);

    checkCuda( cudaMemcpyAsync(dstData, srcData, batchSize*c*h*w*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));

    for (int b = 0; b < batchSize; ++b){
        for(int n = 0; n < num; ++n){
            int index = entry_index(b, n*w*h, 0);
            activationLOGISTICForward(srcData + index, dstData + index, 2*w*h, stream);

            index = entry_index(b, n*w*h, coords);
            activationLOGISTICForward(srcData + index, dstData + index, w*h, stream);
        }
    }

    //softmax start
    int index = entry_index(0, 0, coords + 1);
    softmaxForward(	srcData + index, classes, batchSize*num,
                       (c*h*w)/num,
                       w*h, 1, w*h, 1, dstData + index, stream);

    return 0;
}
#elif NV_TENSORRT_MAJOR == 7
int32_t RegionRT::enqueue(int32_t batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) {
    dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
    dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);

    checkCuda( cudaMemcpyAsync(dstData, srcData, batchSize*c*h*w*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));

    for (int b = 0; b < batchSize; ++b){
        for(int n = 0; n < num; ++n){
            int index = entry_index(b, n*w*h, 0);
            activationLOGISTICForward(srcData + index, dstData + index, 2*w*h, stream);

            index = entry_index(b, n*w*h, coords);
            activationLOGISTICForward(srcData + index, dstData + index, w*h, stream);
        }
    }

    //softmax start
    int index = entry_index(0, 0, coords + 1);
    softmaxForward(srcData + index, classes, batchSize*num,
                   (c*h*w)/num,
                   w*h, 1, w*h, 1, dstData + index, stream);

    return 0;
}
#endif

size_t RegionRT::getSerializationSize() const NOEXCEPT {
    return 6*sizeof(int) + num*2*sizeof(dnnType) + REGIONRT_CLASSNAME_W*classes*sizeof(char);
}

void RegionRT::serialize(void *buffer) const NOEXCEPT {
    char *buf = reinterpret_cast<char*>(buffer),*a=buf;
    writeBUF(buf, classes);
    writeBUF(buf, coords);
    writeBUF(buf, num);
    writeBUF(buf, c);
    writeBUF(buf, h);
    writeBUF(buf, w);
    
    // serialize anchors' coordinates
    for (int i = 0; i < 2*num; i++) {
        writeBUF(buf, bias[i]);
    }

    // save class names
    for (int i = 0; i < classes; i++) {
        char tmp[REGIONRT_CLASSNAME_W];
        strcpy(tmp, classNames[i].c_str());
        for (int j = 0; j < REGIONRT_CLASSNAME_W; j++) {
            writeBUF(buf, tmp[j]);
        }
    }
    
    assert(buf == a + getSerializationSize());
}

const char *RegionRT::getPluginType() const NOEXCEPT {
    return REGIONRT_PLUGIN_NAME;
}

const char *RegionRT::getPluginVersion() const NOEXCEPT {
    return REGIONRT_PLUGIN_VERSION;
}

void RegionRT::destroy() NOEXCEPT { delete this; }

const char *RegionRT::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

void RegionRT::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

bool RegionRT::supportsFormat(DataType type, PluginFormat format) const NOEXCEPT {
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

IPluginV2Ext *RegionRT::clone() const NOEXCEPT {
    auto *p = new RegionRT(classes, coords, num, c, h, w, classNames, bias);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

DataType RegionRT::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const NOEXCEPT {
    return DataType::kFLOAT;
}

void RegionRT::attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                               IGpuAllocator *gpuAllocator) NOEXCEPT {

}

bool RegionRT::isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const NOEXCEPT {
    return false;
}

bool RegionRT::canBroadcastInputAcrossBatch(int inputIndex) const NOEXCEPT {
    return false;
}

void RegionRT::configurePlugin(const Dims *inputDims, int32_t nbInputs, const Dims *outputDims, int32_t nbOutputs,
                               const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast,
                               const bool *outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) NOEXCEPT {

}

void RegionRT::detachFromContext() NOEXCEPT {

}


RegionRTPluginCreator::RegionRTPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void RegionRTPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *RegionRTPluginCreator::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

IPluginV2Ext *RegionRTPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) NOEXCEPT {
    auto *pluginObj = new RegionRT(serialData, serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

IPluginV2Ext *RegionRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT {
    const PluginField *fields = fc->fields;
    assert(fc->nbFields == 6);
    for (int i = 0; i < 6; i++){
        assert(fields[i].type == PluginFieldType::kINT32);
    }
    int classes = *(static_cast<const int*>(fields[0].data));
    int coords = *(static_cast<const int*>(fields[1].data));
    int num = *(static_cast<const int*>(fields[2].data));
    int c = *(static_cast<const int*>(fields[3].data));
    int h = *(static_cast<const int*>(fields[4].data));
    int w = *(static_cast<const int*>(fields[5].data));
    
    std::vector<std::string> classNames(static_cast<const std::string *>(fields[6].data), 
                                        static_cast<const std::string *>(fields[6].data) + fields[6].length);
    std::vector<dnnType> bias_vec(static_cast<const dnnType*>(fields[7].data), 
                                  static_cast<const dnnType*>(fields[7].data) + fields[7].length);
    
    auto *pluginObj = new RegionRT(classes, coords, num, c, h, w, classNames, bias_vec);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char *RegionRTPluginCreator::getPluginName() const NOEXCEPT {
    return REGIONRT_PLUGIN_NAME;
}

const char *RegionRTPluginCreator::getPluginVersion() const NOEXCEPT {
    return REGIONRT_PLUGIN_VERSION;
}

const PluginFieldCollection *RegionRTPluginCreator::getFieldNames() NOEXCEPT {
    return &mFC;
}











