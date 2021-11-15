#ifndef SimpleClassification_H
#define SimpleClassification_H
#include <opencv2/videoio.hpp>
#include "opencv2/opencv.hpp"

#include "ClassificationNN.h"


namespace tk { namespace dnn { 

class SimpleClassification : public ClassificationNN {
private:
    cv::Mat bgr_h;
     
public:

    SimpleClassification() {};
    ~SimpleClassification() {}; 

    tk::dnn::dataDim_t idim;

    bool init(const std::string& tensor_path, const int n_classes=80, 
              const int n_batches=1, const float conf_thresh=0.3);
    
    void preprocess(cv::Mat &frame, const int bi=0);
    
    void postprocess(const int bi=0);
    
    static int max_index(float *arr, int n);

};

} // namespace dnn
} // namespace tk

#endif /* SimpleClassification_H*/


