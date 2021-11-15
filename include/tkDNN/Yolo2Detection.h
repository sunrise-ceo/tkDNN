#ifndef Yolo2Detection_H
#define Yolo2Detection_H
#include <opencv2/videoio.hpp>
#include "opencv2/opencv.hpp"

#include "DetectionNN.h"
#include "DarknetParser.h"

namespace tk { namespace dnn { 

class Yolo2Detection : public DetectionNN {
private:
    int num = 0;
    int coords = 0;
    dnnType* bias_h;
    tk::dnn::RegionInterpret* region_interpret;
    cv::Mat bgr_h;
     
public:

    Yolo2Detection() {};
    ~Yolo2Detection() {}; 

    tk::dnn::dataDim_t idim;

    bool init(const std::string& tensor_path, const std::string& cfg_path, const std::string& names_path, 
              const int n_classes=80, const int n_batches=1, const float conf_thresh=0.3);
    void preprocess(cv::Mat &frame, const int bi=0);
    void postprocess(const int bi=0, const bool mAP=false);
    void draw(std::vector<cv::Mat>& frames) ;
};


} // namespace dnn
} // namespace tk

#endif /* Yolo2Detection_H*/
