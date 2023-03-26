#include <torch/extension.h>
#include "ATen/ATen.h"
#include <iostream>

// generic T either float or fp16 or fp64


void cuda_mm8_three(int N, int M,
                    double *x,
                    double *x1,
                    double *x2,
                    uint8_t *w, int w_stride,
                    uint8_t *w1, int w1_stride,
                    uint8_t *w2, int w2_stride,
                    double *y,
                    double *y1,
                    double *y2,
                    double *r, double *r1, double *r2
                    );
void cuda_mm8_one(int N, int M,
                  double *x,
                  uint8_t *w, int w_stride,
                  double *y,
                    double *r
                    );
void cuda_wkv_forward(int B, int T, int C, double *w, double *u, double *k, double *v, double *y, double *aa, double *bb, double *pp);
void cuda_mm8_three(int N, int M,
                    float *x,
                    float *x1,
                    float *x2,
                    uint8_t *w, int w_stride,
                    uint8_t *w1, int w1_stride,
                    uint8_t *w2, int w2_stride,
                    float *y,
                    float *y1,
                    float *y2,
                    float *r, float *r1, float *r2
                    );
void cuda_mm8_one(int N, int M,
                  float *x,
                  uint8_t *w, int w_stride,
                  float *y,
                    float *r
                    );
void cuda_wkv_forward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y, float *aa, float *bb, float *pp);
void wkv_forward(int64_t B, int64_t T, int64_t C, at::Tensor &w, at::Tensor &u, at::Tensor &k, at::Tensor &v, at::Tensor &y, at::Tensor &aa, at::Tensor &bb, at::Tensor &pp) {
    assert(w.scalar_type() == at::kDouble);
    cuda_wkv_forward(B, T, C, w.data_ptr<double>(), u.data_ptr<double>(), k.data_ptr<double>(), v.data_ptr<double>(), y.data_ptr<double>(), aa.data_ptr<double>(), bb.data_ptr<double>(), pp.data_ptr<double>());
    
}

void mm8_one(int64_t N, int64_t M,
             at::Tensor &x, at::Tensor &w,
             at::Tensor &y,at::Tensor &r) {
    assert(x.stride(0) == 1);
    assert(w.stride(1) == 1);
    assert(y.stride(0) == 1);
    assert(x.scalar_type() == y.scalar_type() && x.scalar_type() == r.scalar_type());

    if( x.scalar_type()== at::kDouble){
        cuda_mm8_one(
        N, M,
        x.data_ptr<double>(),
        w.data_ptr<uint8_t>(), w.stride(0),
        y.data_ptr<double>(),
        r.data_ptr<double>()
        );
    
    }else{
        cuda_mm8_one(
        N, M,
        x.data_ptr<float>(),
        w.data_ptr<uint8_t>(), w.stride(0),
        y.data_ptr<float>(),
        r.data_ptr<float>()
        );
    }

    
}

void mm8_three(int64_t N, int64_t M,
               at::Tensor &x, at::Tensor &x1, at::Tensor &x2,
               at::Tensor &w, at::Tensor &w1, at::Tensor &w2,
               at::Tensor &y, at::Tensor &y1, at::Tensor &y2,
               at::Tensor &r, at::Tensor &r1, at::Tensor &r2) {
    assert(x.stride(0) == 1);
    assert(x1.stride(0) == 1);
    assert(x2.stride(0) == 1);
    assert(w.stride(1) == 1);
    assert(w1.stride(1) == 1);
    assert(w2.stride(1) == 1);
    assert(y.stride(0) == 1);
    assert(y1.stride(0) == 1);
    assert(y2.stride(0) == 1);
    assert(x.scalar_type() == y.scalar_type() && x.scalar_type() == r.scalar_type());
    if(x.scalar_type() == at::kDouble){
        cuda_mm8_three(
        N, M,
        x.data_ptr<double>(),
        x1.data_ptr<double>(),
        x2.data_ptr<double>(),
        w.data_ptr<uint8_t>(), w.stride(0),
        w1.data_ptr<uint8_t>(), w1.stride(0),
        w2.data_ptr<uint8_t>(), w2.stride(0),
        y.data_ptr<double>(),
        y1.data_ptr<double>(),
        y2.data_ptr<double>(),
        r.data_ptr<double>(), r1.data_ptr<double>(), r2.data_ptr<double>()
        );
    }
    else{
        cuda_mm8_three(
        N, M,
        x.data_ptr<float>(),
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        w.data_ptr<uint8_t>(), w.stride(0),
        w1.data_ptr<uint8_t>(), w1.stride(0),
        w2.data_ptr<uint8_t>(), w2.stride(0),
        y.data_ptr<float>(),
        y1.data_ptr<float>(),
        y2.data_ptr<float>(),
        r.data_ptr<float>(), r1.data_ptr<float>(), r2.data_ptr<float>()
        );}
    
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("wkv_forward", &wkv_forward, "wkv forward");
//     m.def("mm8_one", &mm8_one, "mm8 one");
//     m.def("mm8_three", &mm8_three, "mm8 three");
// }

TORCH_LIBRARY(rwkv, m) {
    m.def("wkv_forward", wkv_forward);
    m.def("mm8_one", mm8_one);
    m.def("mm8_three", mm8_three);
}
