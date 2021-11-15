#include "Yolo2Detection.h"
#include "NvInfer.h"

namespace tk { namespace dnn {


bool Yolo2Detection::init(const std::string& tensor_path, const std::string& cfg_path, const std::string& names_path,
                          const int n_classes, const int n_batches, const float conf_thresh) {
    
    //convert network to tensorRT
    std::cout<<(tensor_path).c_str()<<"\n";
    netRT = new tk::dnn::NetworkRT(NULL, (tensor_path).c_str());
    
    // find Region layer's line number in Darknet .cfg file 
    int regionLineNo = noRegionLine(cfg_path);
  
    // read Region layer parameters from Darknet .cfg file 
    std::vector<float> anchorsTemp;
    int classesTemp, coordsTemp, numTemp;  
    loadRegionInfo(cfg_path, regionLineNo, anchorsTemp, numTemp, classesTemp, coordsTemp);

    classes = classesTemp;
    coords = coordsTemp;
    num = numTemp;
    bias_h = new dnnType[num*2]; 
    memcpy(bias_h, anchorsTemp.data(), sizeof(dnnType)*num*2);
    
    classesNames = darknetReadNames(names_path);  
  
    nBatches = n_batches;
    confThreshold = conf_thresh;
    idim = netRT->input_dim;    
    idim.n = nBatches;
    
    region_interpret = new tk::dnn::RegionInterpret(netRT->output_dim, netRT->output_dim, 
                                                    n_classes, coords, num, 
                                                    conf_thresh, bias_h);

#ifndef OPENCV_CUDACONTRIB
    checkCuda(cudaMallocHost(&input, sizeof(dnnType)*idim.tot()));
#endif
    checkCuda(cudaMalloc(&input_d, sizeof(dnnType)*idim.tot()));

    // class colors precompute    
    for(int c=0; c<classes; c++) {
        int offset = c*123457 % classes;
        float r = getColor(2, offset, classes);
        float g = getColor(1, offset, classes);
        float b = getColor(0, offset, classes);
        colors[c] = cv::Scalar(int(255.0*b), int(255.0*g), int(255.0*r));
    }

    return true;
} 


void Yolo2Detection::preprocess(cv::Mat &frame, const int bi){
#ifdef OPENCV_CUDACONTRIB
    cv::cuda::GpuMat orig_img, img_resized;
    orig_img = cv::cuda::GpuMat(frame);
    cv::cuda::resize(orig_img, img_resized, cv::Size(netRT->input_dim.w, netRT->input_dim.h));

    img_resized.convertTo(imagePreproc, CV_32FC3, 1/255.0); 

    //split channels
    cv::cuda::split(imagePreproc,bgr);//split source

    //write channels
    for(int i=0; i<netRT->input_dim.c; i++) {
        int size = imagePreproc.rows * imagePreproc.cols;
        int ch = netRT->input_dim.c-1 -i;
        bgr[ch].download(bgr_h); //TODO: don't copy back on CPU
        checkCuda( cudaMemcpy(input_d + i*size + netRT->input_dim.tot()*bi, (float*)bgr_h.data, size*sizeof(dnnType), cudaMemcpyHostToDevice));
    }
#else
    cv::resize(frame, frame, cv::Size(netRT->input_dim.w, netRT->input_dim.h));
    frame.convertTo(imagePreproc, CV_32FC3, 1/255.0); 

    //split channels
    cv::split(imagePreproc,bgr);//split source

    //write channels
    for(int i=0; i<netRT->input_dim.c; i++) {
        int idx = i*imagePreproc.rows*imagePreproc.cols;
        int ch = netRT->input_dim.c-1 -i;
        memcpy((void*)&input[idx + netRT->input_dim.tot()*bi], (void*)bgr[ch].data, imagePreproc.rows*imagePreproc.cols*sizeof(dnnType));     
    }
    checkCuda(cudaMemcpyAsync(input_d + netRT->input_dim.tot()*bi, 
                              input + netRT->input_dim.tot()*bi, 
                              netRT->input_dim.tot()*sizeof(dnnType), 
                              cudaMemcpyHostToDevice, netRT->stream));
#endif
}

    
void Yolo2Detection::postprocess(const int bi, const bool mAP) {
    
    dnnType* rt_out = new dnnType[netRT->output_dim.tot()]; 
    checkCuda(cudaMemcpy(rt_out, netRT->output, 
                         netRT->output_dim.tot()*sizeof(dnnType), 
                         cudaMemcpyDeviceToHost));
    
    region_interpret->interpretData(rt_out, originalSize[bi].width, originalSize[bi].height);
    
    // fill detected
    detected.clear();
    for (int i = 0; i < region_interpret->res_boxes_n; i++) {
        detected.push_back(region_interpret->res_boxes[i]);
    }
    
    batchDetected.push_back(detected);
}


/**
 * Method to draw bounding boxes and labels on a frame.
 * 
 * @param frames original frame to draw bounding box on.
 */
void Yolo2Detection::draw(std::vector<cv::Mat>& frames) {
    tk::dnn::box b;
    int x0, w, x1, y0, h, y1;
    int objClass;
    std::string det_class;
    int baseline = 0;
    float font_scale = 0.5;
    int thickness = 2;   

    for(int bi=0; bi<frames.size(); ++bi){
        // draw dets
        for(int i=0; i<batchDetected[bi].size(); i++) { 
            b           = batchDetected[bi][i];
            x0   		= b.x - b.w/2;
            x1   		= b.x + b.w/2;
            y0   		= b.y - b.h/2;
            y1   		= b.y + b.h/2;
            det_class 	= classesNames[b.cl];

            // draw rectangle
            cv::rectangle(frames[bi], cv::Point(x0, y0), cv::Point(x1, y1), colors[b.cl], 2); 

            // draw label
            cv::Size text_size = getTextSize(det_class, cv::FONT_HERSHEY_SIMPLEX, 
                                             font_scale, thickness, &baseline);
            cv::rectangle(frames[bi], cv::Point(x0, y0), 
                          cv::Point((x0 + text_size.width - 2), (y0 - text_size.height - 2)), 
                          colors[b.cl], -1);                      
            cv::putText(frames[bi], det_class, 
                        cv::Point(x0, (y0 - (baseline / 2))), cv::FONT_HERSHEY_SIMPLEX, 
                        font_scale, cv::Scalar(255, 255, 255), thickness);
        }
    }
}
    
    
}}
