#ifndef CLASSIFICATIONNN_H
#define CLASSIFICATIONNN_H

#include <iostream>
#include <signal.h>
#include <stdlib.h>    
#ifdef __linux__
#include <unistd.h>
#endif 

#include <mutex>
#include "utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "tkdnn.h"

//#define OPENCV_CUDACONTRIB //if OPENCV has been compiled with CUDA and contrib.

#ifdef OPENCV_CUDACONTRIB
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#endif


namespace tk { namespace dnn {

class ClassificationNN {

    protected:
        tk::dnn::NetworkRT *netRT = nullptr;
        dnnType *input_d;
        int nBatches = 1;

#ifdef OPENCV_CUDACONTRIB
        cv::cuda::GpuMat bgr[3];
        cv::cuda::GpuMat imagePreproc;
#else
        cv::Mat bgr[3];
        cv::Mat imagePreproc;
        dnnType *input;
#endif

        /**
         * This method preprocess the image, before feeding it to the NN.
         *
         * @param frame original frame to adapt for inference.
         * @param bi batch index
         */
        virtual void preprocess(cv::Mat &frame, const int bi=0) = 0;

        /**
         * This method postprocess the output of the NN to obtain the correct 
         * class scores. 
         * 
         * @param bi batch index
         */
        virtual void postprocess(const int bi=0) = 0;

    public:
        int classes = 0;
        float confThreshold = 0.3; /*threshold on the confidence of the boxes*/

        std::vector<dnnType*> batchClassified; /*class scores in the net's output*/
        std::vector<double> stats; /*keeps track of inference times (ms)*/
        std::vector<std::string> classesNames;

        ClassificationNN() {};
        ~ClassificationNN(){};

        /**
         * Method used to initialize the class, allocate memory and compute 
         * needed data.
         * 
         * @param tensor_path path to the rt file of the NN.
         * @param n_classes number of classes for the given dataset.
         * @param n_batches maximum number of batches to use in inference
         * @return true if everything is correct, false otherwise.
         */
        virtual bool init(const std::string& tensor_path, const int n_classes=80, 
                          const int n_batches=1, const float conf_thresh=0.3) = 0;
        
        /**
         * This method performs the whole classification of the NN.
         * 
         * @param frames frames to run detection on.
         * @param cur_batches number of batches to use in inference
         * @param save_times if set to true, preprocess, inference and postprocess times 
         *        are saved on a csv file, otherwise not.
         * @param times pointer to the output stream where to write times
         */
        void update(std::vector<cv::Mat>& frames, const int cur_batches=1, 
                    bool save_times=false, std::ofstream *times=nullptr) {
            
            if (save_times && times==nullptr)
                FatalError("save_times set to true, but no valid ofstream given");
            if (cur_batches > nBatches)
                FatalError("A batch size greater than nBatches cannot be used");

            if (TKDNN_VERBOSE) printCenteredTitle(" TENSORRT classification ", '=', 30); 
            {
                TKDNN_TSTART
                for (int bi = 0; bi < cur_batches; ++bi) {
                    if(!frames[bi].data)
                        FatalError("No image data to feed to classification!");
                    preprocess(frames[bi], bi);    
                }
                TKDNN_TSTOP
                if (save_times) *times << t_ns <<";";
            }

            //do inference
            tk::dnn::dataDim_t dim = netRT->input_dim;
            dim.n = cur_batches;
            {
                if (TKDNN_VERBOSE) dim.print();
                TKDNN_TSTART
                    
                dnnType* output = netRT->infer(dim, input_d);
                netRT->output = output;
                
                TKDNN_TSTOP
                if (TKDNN_VERBOSE) dim.print();
                stats.push_back(t_ns);
                if (save_times) *times << t_ns << ";";
            }

            batchClassified.clear();
            {
                TKDNN_TSTART
                for (int bi = 0; bi < cur_batches; ++bi)
                    postprocess(bi);
                TKDNN_TSTOP
                if (save_times) *times << t_ns << "\n";
            }
        }      
};

}}

#endif /* CLASSIFICATIONNN_H*/
