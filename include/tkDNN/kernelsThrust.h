#ifndef KERNELSTHRUST_H
#define KERNELSTHRUST_H
#define STEREO_SCALE_FACTOR 5.4

#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/gather.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>


#include "tkdnn.h"

struct threshold : public thrust::binary_function<float,float,float>
{
  __host__ __device__
  float operator()(float x, float y) { 
    double toll = 1e-6; 
    if(fabsf(x-y)>toll) 
        return 0.0f; 
    else
        return x; 
    }
};

void sort(dnnType *src_begin, dnnType *src_end, int *idsrc);
void topk(dnnType *src_begin, int *idsrc, int K, float *topk_scores,
            int *topk_inds, float *topk_ys, float *topk_xs);
// void sortAndTopKonDevice(dnnType *src_begin, int *idsrc, float *topk_scores, int *topk_inds, float *topk_ys, float *topk_xs, const int size, const int K, const int n_classes);            
void normalize(float *bgr, const int ch, const int h, const int w, const float *mean, const float *stddev);
void transformDep(float *src_begin, float *src_end, float *dst_begin, float *dst_end);
void subtractWithThreshold(dnnType *src_begin, dnnType *src_end, dnnType *src2_begin, dnnType *src_out, struct threshold op);
void topKxyclasses(int *ids_begin, int *ids_end, const int K, const int size, const int wh, int *clses, int *xs, int *ys);
void topKxyAddOffset(int * ids_begin, const int K, const int size, int *intxs_begin, int *intys_begin, 
                     float *xs_begin, float *ys_begin, dnnType *src_begin, float *src_out, int *ids_out);
void bboxes(int * ids_begin, const int K, const int size, float *xs_begin, float *ys_begin, 
            dnnType *src_begin, float *bbx0, float *bbx1, float *bby0, float *bby1, float *src_out, int *ids_out);
void getRecordsFromTopKId(int * ids_begin, const int K, const int ch, const int size, dnnType *src_begin, float *src_out, int *ids_out);

void maxElem(dnnType *src_begin, dnnType *dst_begin, const int c, const int h, const int w);

void metricDepth(dnnType *src,dnnType *dst,int64_t min_depth,int64_t max_depth,int64_t c,int64_t h,int64_t w);


#endif //KERNELSTHRUST_H