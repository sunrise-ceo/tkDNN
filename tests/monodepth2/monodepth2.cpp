#include <iostream>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <tkdnn.h>

const char* encoder_conv1_bin = "monodepth2/layers/encoder/encoder-conv1.bin";
const char* encoder_layer1_bin[] = {
        "monodepth2/layers/encoder/encoder-layer1-0-conv1.bin",
        "monodepth2/layers/encoder/encoder-layer1-0-conv2.bin",
        "monodepth2/layers/encoder/encoder-layer1-1-conv1.bin",
        "monodepth2/layers/encoder/encoder-layer1-1-conv2.bin",
};

const char* encoder_layer2_bin[] = {
        "monodepth2/layers/encoder/encoder-layer2-0-conv1.bin",
        "monodepth2/layers/encoder/encoder-layer2-0-conv2.bin",
        "monodepth2/layers/encoder/encoder-layer2-0-downsample-0.bin",
        "monodepth2/layers/encoder/encoder-layer2-1-conv1.bin",
        "monodepth2/layers/encoder/encoder-layer2-1-conv2.bin"
};

const char* encoder_layer3_bin[]={
        "monodepth2/layers/encoder/encoder-layer3-0-conv1.bin",
        "monodepth2/layers/encoder/encoder-layer3-0-conv2.bin",
        "monodepth2/layers/encoder/encoder-layer3-0-downsample-0.bin",
        "monodepth2/layers/encoder/encoder-layer3-1-conv1.bin",
        "monodepth2/layers/encoder/encoder-layer3-1-conv2.bin"
};

const char* encoder_layer4_bin[] = {
        "monodepth2/layers/encoder/encoder-layer4-0-conv1.bin",
        "monodepth2/layers/encoder/encoder-layer4-0-conv2.bin",
        "monodepth2/layers/encoder/encoder-layer4-0-downsample-0.bin",
        "monodepth2/layers/encoder/encoder-layer4-1-conv1.bin",
        "monodepth2/layers/encoder/encoder-layer4-1-conv2.bin"
};

const char *encoder_fc_bin = "monodepth2/layers/encoder/encoder-fc.bin";

const char* decoder_layer_bin[] = {
        "monodepth2/layers/depth_decoder/decoder-0-conv-conv.bin",
        "monodepth2/layers/depth_decoder/decoder-1-conv-conv.bin",
        "monodepth2/layers/depth_decoder/decoder-2-conv-conv.bin",
        "monodepth2/layers/depth_decoder/decoder-3-conv-conv.bin",
        "monodepth2/layers/depth_decoder/decoder-4-conv-conv.bin",
        "monodepth2/layers/depth_decoder/decoder-5-conv-conv.bin",
        "monodepth2/layers/depth_decoder/decoder-6-conv-conv.bin",
        "monodepth2/layers/depth_decoder/decoder-7-conv-conv.bin",
        "monodepth2/layers/depth_decoder/decoder-8-conv-conv.bin",
        "monodepth2/layers/depth_decoder/decoder-9-conv-conv.bin"
};

const char* decoder_dispconv_layer_bin[] = {
        "monodepth2/layers/depth_decoder/decoder-10-conv.bin ",
        "monodepth2/layers/depth_decoder/decoder-11-conv.bin ",
        "monodepth2/layers/depth_decoder/decoder-12-conv.bin ",
        "monodepth2/layers/depth_decoder/decoder-13-conv.bin "
};

const char* output_bin[] = {
        "monodepth2/debug/outputs/output-disp-0.bin",
        "monodepth2/debug/outputs/output-disp-1.bin",
        "monodepth2/debug/outputs/output-disp-2.bin",
        "monodepth2/debug/outputs/output-disp-3.bin"
};

const char* input_monodepth2_bin[] = {"monodepth2/debug/input.bin"};


int main(){

    tk::dnn::dataDim_t dim(0,3,192,640,1);
    tk::dnn::Network net(dim);
    std::vector<tk::dnn::Layer*> features;
    new tk::dnn::Conv2d(&net,64,7,7,2,2,3,3,encoder_conv1_bin, true);
    tk::dnn::Layer *encoder_relu_1 =  new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
    features.push_back(encoder_relu_1);
    tk::dnn::Layer *last = new tk::dnn::Pooling(&net,3,3,2,2,1,1,tk::dnn::POOLING_MAX);

    //layer 1
    for(int i=0;i<4;i+2){
        new tk::dnn::Conv2d(&net,64,3,3,1,1,1,1,encoder_layer1_bin[i], true);
        new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
        new tk::dnn::Conv2d(&net,64,3,3,1,1,1,1,encoder_layer1_bin[i+1],true);
        new tk::dnn::Shortcut(&net,last);
        last = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
    }
    features.push_back(last);

    //layer2
    for(int i=0;i<5;i+3){
        new tk::dnn::Conv2d(&net,128,3,3,2,2,1,1,encoder_layer2_bin[i],true);
        new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
        tk::dnn::Layer *bn2 = new tk::dnn::Conv2d(&net,128,3,3,1,1,1,1,encoder_layer2_bin[i+1],true);
        if(i == 0){
            new tk::dnn::Route(&net,&last,1);
            new tk::dnn::Conv2d(&net,128,1,1,2,2,0,0,encoder_layer2_bin[i+2], true);
            new tk::dnn::Shortcut(&net,bn2);
        }else{
            new tk::dnn::Shortcut(&net,last);
        }
        last = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
    }
    features.push_back(last);

    //layer 3

    for(int i=0;i<5;i+3){
        new tk::dnn::Conv2d(&net,256,3,3,2,2,1,1,encoder_layer3_bin[i], true);
        new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
        tk::dnn::Layer* bn2 = new tk::dnn::Conv2d(&net,256,3,3,1,1,1,1,encoder_layer3_bin[i+1],true);
        if (i == 0){
            new tk::dnn::Route(&net,&last,1);
            new tk::dnn::Conv2d(&net,256,1,1,2,2,0,0,encoder_layer3_bin[i+2], true);
            new tk::dnn::Shortcut(&net,last);
        }
        else{
            new tk::dnn::Shortcut(&net,last);
        }
        last = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
    }
    features.push_back(last);

    //layer4
    for(int i=0;i<5;i+3){
        new tk::dnn::Conv2d(&net,512,3,3,2,2,1,1,encoder_layer4_bin[i], true);
        new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
        tk::dnn::Layer* bn2 = new tk::dnn::Conv2d(&net,512,3,3,1,1,1,1,encoder_layer4_bin[i+1], true);
        if(i == 0){
            new tk::dnn::Route(&net,&last,1);
            new tk::dnn::Conv2d(&net,512,1,1,2,2,0,0,encoder_layer4_bin[i+2], true);
            new tk::dnn::Shortcut(&net,last);
        }else{
            new tk::dnn::Shortcut(&net,last);
        }
        last = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
    }
    features.push_back(last);
    new tk::dnn::Pooling(&net,1,1,1,1,0,0,tk::dnn::POOLING_AVERAGE);
    last = new tk::dnn::Dense(&net,100,encoder_fc_bin);


    //decoders
    new tk::dnn::Route(&net,&features[4],1);

    std::vector<tk::dnn::Layer*> depth_conv_features;
    new tk::dnn::ReflectionPAD2D(&net,1);
    new tk::dnn::Conv2d(&net,256,3,3,1,1,0,0,decoder_layer_bin[0], false);
    new tk::dnn::Activation(&net,CUDNN_ACTIVATION_ELU);
    new tk::dnn::Upsample(&net,2);
    new tk::dnn::Route(&net,&features[3],1);
    new tk::dnn::ReflectionPAD2D(&net,1);
    new tk::dnn::Conv2d(&net,256,3,3,1,1,0,0,decoder_layer_bin[1], false);

    new tk::dnn::ReflectionPAD2D(&net,1);
    new tk::dnn::Conv2d(&net,128,3,3,1,1,0,0,decoder_layer_bin[2], false);
    new tk::dnn::Activation(&net,CUDNN_ACTIVATION_ELU);
    new tk::dnn::Upsample(&net,2);
    new tk::dnn::Route(&net,&features[2],1);
    new tk::dnn::ReflectionPAD2D(&net,1);
    new tk::dnn::Conv2d(&net,128,3,3,1,1,0,0,decoder_layer_bin[3], false);
    last = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_ELU);
    depth_conv_features.push_back(last);

    new tk::dnn::ReflectionPAD2D(&net,1);
    new tk::dnn::Conv2d(&net,64,3,3,1,1,0,0,decoder_layer_bin[4],false);
    new tk::dnn::Activation(&net,CUDNN_ACTIVATION_ELU);
    new tk::dnn::Upsample(&net,2);
    new tk::dnn::Route(&net,&features[1],1);
    new tk::dnn::ReflectionPAD2D(&net,1);
    new tk::dnn::Conv2d(&net,64,3,3,1,1,0,0,decoder_layer_bin[5], false);
    last = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_ELU);
    depth_conv_features.push_back(last);

    new tk::dnn::ReflectionPAD2D(&net,1);
    new tk::dnn::Conv2d(&net,32,3,3,1,1,0,0,decoder_layer_bin[6], false);
    new tk::dnn::Activation(&net,CUDNN_ACTIVATION_ELU);
    new tk::dnn::Upsample(&net,2);
    new tk::dnn::Route(&net,&features[0],1);
    new tk::dnn::ReflectionPAD2D(&net,1);
    new tk::dnn::Conv2d(&net,32,3,3,1,1,0,0,decoder_layer_bin[7], false);
    last = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_ELU);
    depth_conv_features.push_back(last);

    new tk::dnn::ReflectionPAD2D(&net,1);
    new tk::dnn::Conv2d(&net,16,3,3,1,1,0,0,decoder_layer_bin[8], false);
    new tk::dnn::Activation(&net,CUDNN_ACTIVATION_ELU);
    new tk::dnn::Upsample(&net,2);
    new tk::dnn::ReflectionPAD2D(&net,1);
    new tk::dnn::Conv2d(&net,16,3,3,1,1,0,0,decoder_layer_bin[9], false);
    last = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_ELU);
    depth_conv_features.push_back(last);


    new tk::dnn::Route(&net,&depth_conv_features[3],1);
    new tk::dnn::ReflectionPAD2D(&net,1);
    new tk::dnn::Conv2d(&net,1,3,3,1,1,0,0,decoder_dispconv_layer_bin[0], false);
    tk::dnn::Layer *disp0 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_SIGMOID);
    disp0->setFinal();

    new tk::dnn::Route(&net,&depth_conv_features[2],1);
    new tk::dnn::ReflectionPAD2D(&net,1);
    new tk::dnn::Conv2d(&net,1,3,3,1,1,0,0,decoder_dispconv_layer_bin[1],false);
    tk::dnn::Layer *disp1 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_SIGMOID);
    disp1->setFinal();


    new tk::dnn::Route(&net,&depth_conv_features[1],1);
    new tk::dnn::ReflectionPAD2D(&net,1);
    new tk::dnn::Conv2d(&net,1,3,3,1,1,0,0,decoder_dispconv_layer_bin[2], false);
    tk::dnn::Layer *disp2 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_SIGMOID);
    disp2->setFinal();

    new tk::dnn::Route(&net,&depth_conv_features[0],1);
    new tk::dnn::ReflectionPAD2D(&net,1);
    new tk::dnn::Conv2d(&net,1,3,3,1,1,0,0,decoder_dispconv_layer_bin[3], false);
    tk::dnn::Layer *disp3 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_SIGMOID);
    disp3->setFinal();


    dnnType *data;
    dnnType *input_H;
    readBinaryFile(input_monodepth2_bin[0],dim.tot(),&input_H,&data);
    std::cout<<"Input:"<<std::endl;

    net.print();

    tk::dnn::NetworkRT netRT(&net, net.getNetworkRTName("monodepth2"));
    tk::dnn::dataDim_t dim1 = dim;
    dnnType *cudnn_out = nullptr;
    printCenteredTitle(" CUDNN inference ", '=', 30);
    {
        dim1.print();
        TKDNN_TSTART
        net.infer(dim1, data);
        TKDNN_TSTOP
        dim1.print();
    }

    tk::dnn::dataDim_t dim2 = dim;
    printCenteredTitle(" TENSORRT inference ", '=', 30);
    {
        dim2.print();
        TKDNN_TSTART
        netRT.infer(dim2, data);
        TKDNN_TSTOP
        dim2.print();
    }
    tk::dnn::Layer *outs[4] = {disp0,disp1,disp2,disp3};
    int ret_cudnn = 0, ret_tensorrt = 0, ret_cudnn_tensorrt = 0;
    for(int i=0;i<4;i++){
        printCenteredTitle((std::string("MONODEPTH2 CHECK RESULTS ") + std::to_string(i) + " ").c_str(), '=', 30);
        outs[i]->output_dim.print();

        dnnType *out, *out_h;
        int odim = outs[i]->output_dim.tot();
        readBinaryFile(output_bin[i], odim, &out_h, &out);

        dnnType *cudnn_out, *rt_out;
        cudnn_out = outs[i]->dstData;
        rt_out = (dnnType *)netRT.buffersRT[i];
        std::cout<<"CUDNN vs correct";
        ret_cudnn |= checkResult(odim, cudnn_out, out) == 0 ? 0: ERROR_CUDNN;
        std::cout<<"TRT   vs correct";
        ret_tensorrt |= checkResult(odim, rt_out, out) == 0 ? 0 : ERROR_TENSORRT;
        std::cout<<"CUDNN vs TRT    ";
        ret_cudnn_tensorrt |= checkResult(odim, cudnn_out, rt_out) == 0 ? 0 : ERROR_CUDNNvsTENSORRT;
    }


    return ret_cudnn | ret_tensorrt | ret_cudnn_tensorrt;
}