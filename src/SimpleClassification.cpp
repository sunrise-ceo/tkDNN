#include "NvInfer.h"
#include "SimpleClassification.h"


namespace tk { namespace dnn {


bool SimpleClassification::init(const std::string& tensor_path, const int n_classes, 
                                const int n_batches, const float conf_thresh) {
    
    //convert network to tensorRT
    std::cout << (tensor_path).c_str() << "\n";
    netRT = new tk::dnn::NetworkRT(NULL, (tensor_path).c_str() );
    
    classes = n_classes;
    nBatches = n_batches;
    confThreshold = conf_thresh;
    idim = netRT->input_dim;    
    idim.n = nBatches;
    
#ifndef OPENCV_CUDACONTRIB
    checkCuda(cudaMallocHost(&input, sizeof(dnnType) * idim.tot()));
#endif
    checkCuda(cudaMalloc(&input_d, sizeof(dnnType) * idim.tot()));

    return true;
} 


void SimpleClassification::preprocess(cv::Mat &frame, const int bi){
#ifdef OPENCV_CUDACONTRIB
    cv::cuda::GpuMat orig_img, img_resized;
    orig_img = cv::cuda::GpuMat(frame);
    cv::cuda::resize(orig_img, img_resized, cv::Size(netRT->input_dim.w, netRT->input_dim.h));

    img_resized.convertTo(imagePreproc, CV_32FC3, 1/255.0); 

    //split channels
    cv::cuda::split(imagePreproc,bgr);//split source

    //write channels
    for (int i = 0; i < netRT->input_dim.c; i++) {
        int size = imagePreproc.rows * imagePreproc.cols;
        int ch = netRT->input_dim.c - 1 - i;
        bgr[ch].download(bgr_h); //TODO: don't copy back on CPU
        checkCuda(cudaMemcpy(input_d + (i *size) + (netRT->input_dim.tot() * bi), 
                             (float*) bgr_h.data, size * sizeof(dnnType), 
                             cudaMemcpyHostToDevice));
    }
#else
    cv::resize(frame, frame, cv::Size(netRT->input_dim.w, netRT->input_dim.h));
    frame.convertTo(imagePreproc, CV_32FC3, 1/255.0); 

    //split channels
    cv::split(imagePreproc,bgr);//split source

    //write channels
    for(int i = 0; i < netRT->input_dim.c; i++) {
        int idx = i * imagePreproc.rows * imagePreproc.cols;
        int ch = netRT->input_dim.c-1 -i;
        memcpy((void*) &input[idx + netRT->input_dim.tot()*bi], 
               (void*) bgr[ch].data, 
               imagePreproc.rows * imagePreproc.cols * sizeof(dnnType));     
    }
    checkCuda(cudaMemcpyAsync(input_d + (netRT->input_dim.tot() * bi), 
                              input + (netRT->input_dim.tot() * bi), 
                              netRT->input_dim.tot() * sizeof(dnnType), 
                              cudaMemcpyHostToDevice, netRT->stream));
#endif
}

    
void SimpleClassification::postprocess(const int bi) {
    
    int batch_len = netRT->output_dim.w * netRT->output_dim.h * netRT->output_dim.c;  
    dnnType* rt_out = new dnnType[batch_len];
    
    checkCuda(cudaMemcpy(rt_out, 
                         netRT->output + (batch_len * bi), 
                         batch_len * sizeof(dnnType), 
                         cudaMemcpyDeviceToHost));
    
    batchClassified.push_back(rt_out);
}
    
    
int SimpleClassification::max_index(float *arr, int n) {
    if (n <= 0) return -1;
    int i, max_i = 0;
    float max = arr[0];
    for (i = 1; i < n; ++i) {
        if (arr[i] > max) {
            max = arr[i];
            max_i = i;
        }
    }
    return max_i;
}
   
    
}}
