// CoViAR python data loader.
// Part of this implementation is modified from the tutorial at
// https://blog.csdn.net/leixiaohua1020/article/details/50618190
// and FFmpeg extract_mv example.


#include <Python.h>
#include "numpy/arrayobject.h"

#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <libavutil/motion_vector.h>
#include <libavformat/avformat.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
#include <libavcodec/avcodec.h>
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"


#define FF_INPUT_BUFFER_PADDING_SIZE 32
#define MV 1
#define RESIDUAL 2

static const char *filename = NULL;


static PyObject *CoviarError;

void *strrp(char *src,char *sub,char *rp,char *p) {
    int sub_len=strlen(sub);
    char *po=NULL,*q=src;
    while((po=strstr(q,sub))!=NULL) {
        strncat(p,q,po-q);
        strcat(p,rp);
        q+=po-q+sub_len;
    }
    strcat(p,q);
}

void create_and_load_bgr(AVFrame *pFrame, AVFrame *pFrameBGR, uint8_t *buffer,
    PyArrayObject ** arr, int cur_pos, int begin_pos) {

    int numBytes = avpicture_get_size(AV_PIX_FMT_BGR24, pFrame->width, pFrame->height);
    buffer = (uint8_t*) av_malloc(numBytes * sizeof(uint8_t));
    avpicture_fill((AVPicture*) pFrameBGR, buffer, AV_PIX_FMT_BGR24, pFrame->width, pFrame->height);

    struct SwsContext *img_convert_ctx;
    img_convert_ctx = sws_getCachedContext(NULL,
        pFrame->width, pFrame->height, AV_PIX_FMT_YUV420P,
        pFrame->width, pFrame->height, AV_PIX_FMT_BGR24,
        SWS_BICUBIC, NULL, NULL, NULL);

    sws_scale(img_convert_ctx,
        pFrame->data,
        pFrame->linesize, 0, pFrame->height,
        pFrameBGR->data,
        pFrameBGR->linesize);
    sws_freeContext(img_convert_ctx);

    int linesize = pFrame->width * 3;
    int height = pFrame->height;

    int stride_0 = height * linesize;
    int stride_1 = linesize;
    int stride_2 = 3;

    uint8_t *src  = (uint8_t*) pFrameBGR->data[0];
    uint8_t *dest = (uint8_t*) (*arr)->data;

    if (cur_pos == begin_pos) {
        memcpy(dest, src, height * linesize * sizeof(uint8_t));
        memcpy(dest + stride_0, src, height * linesize * sizeof(uint8_t));
    }else {
        memcpy(dest, dest + stride_0, height * linesize * sizeof(uint8_t));
        memcpy(dest + stride_0, src, height * linesize * sizeof(uint8_t));
    }
    av_free(buffer);
}


void create_and_load_mv_residual(
    AVFrameSideData *sd,
    PyArrayObject * bgr_arr,
    PyArrayObject * mv_arr,
    PyArrayObject * res_arr,
    int begin_pos,
    int cur_pos,
    int accumulate,
    int representation,
    int *accu_src,
    int *accu_src_old,
    int width,
    int height) {

    int p_dst_x, p_dst_y, p_src_x, p_src_y, val_x, val_y;
    const AVMotionVector *mvs = (const AVMotionVector *)sd->data;

    for (int i = 0; i < sd->size / sizeof(*mvs); i++) {
        const AVMotionVector *mv = &mvs[i];
        assert(mv->source == -1);

        if (mv->dst_x - mv->src_x != 0 || mv->dst_y - mv->src_y != 0) {

            val_x = mv->dst_x - mv->src_x;
            val_y = mv->dst_y - mv->src_y;

            for (int x_start = (-1 * mv->w / 2); x_start < mv->w / 2; ++x_start) {
                for (int y_start = (-1 * mv->h / 2); y_start < mv->h / 2; ++y_start) {
                    p_dst_x = mv->dst_x + x_start;
                    p_dst_y = mv->dst_y + y_start;

                    p_src_x = mv->src_x + x_start;
                    p_src_y = mv->src_y + y_start;

                    if (p_dst_y >= 0 && p_dst_y < height &&
                        p_dst_x >= 0 && p_dst_x < width &&
                        p_src_y >= 0 && p_src_y < height &&
                        p_src_x >= 0 && p_src_x < width) {

                        // Write MV.
                        if (accumulate) {
                            for (int c = 0; c < 2; ++c) {
                                accu_src       [p_dst_x * height * 2 + p_dst_y * 2 + c]
                                 = accu_src_old[p_src_x * height * 2 + p_src_y * 2 + c];
                            }
                        } else {
                            *((int32_t*)PyArray_GETPTR4(mv_arr, cur_pos - begin_pos,
                                        p_dst_y, p_dst_x, 0)) = val_x;
                            *((int32_t*)PyArray_GETPTR4(mv_arr, cur_pos - begin_pos,
                                        p_dst_y, p_dst_x, 1)) = val_y;
                        }
                    }
                }
            }
        }
    }
    if (accumulate) {
        memcpy(accu_src_old, accu_src, width * height * 2 * sizeof(int));
    }
    if (cur_pos > 0){
        if (accumulate) {
            if (representation == MV) {
                for (int x = 0; x < height; ++x) {
                    for (int y = 0; y < width; ++y) {
                        *((int32_t*)PyArray_GETPTR4(mv_arr, cur_pos - begin_pos, x, y, 0))
                         = y - accu_src[y * height * 2 + x * 2];
                        *((int32_t*)PyArray_GETPTR4(mv_arr, cur_pos - begin_pos, x, y, 1))
                         = x - accu_src[y * height * 2 + x * 2 + 1];
                    }
                }
            }
        }

        if (representation == RESIDUAL) {

            uint8_t *bgr_data = (uint8_t*) bgr_arr->data;
            int32_t *res_data = (int32_t*) res_arr->data;

            int stride_0 = height * width * 3;
            int stride_1 = width * 3;
            int stride_2 = 3;

            int y;

            int offset = (cur_pos - begin_pos)*height*width*3;
            for (y = 0; y < height; ++y) {
                int c, x, src_x, src_y, location, location2, location_src;
                int32_t tmp;
                for (x = 0; x < width; ++x) {
                    tmp = x * height * 2 + y * 2;
                    if (accumulate) {
                        src_x = accu_src[tmp];
                        src_y = accu_src[tmp + 1];
                    } else {
                        src_x = x - (*((int32_t*)PyArray_GETPTR4(mv_arr, cur_pos - begin_pos, y, x, 0)));
                        src_y = y - (*((int32_t*)PyArray_GETPTR4(mv_arr, cur_pos - begin_pos, y, x, 1)));
                    }
                    location_src = src_y * stride_1 + src_x * stride_2;

                    location = y * stride_1 + x * stride_2;
                    for (c = 0; c < 3; ++c) {
                        location2 = stride_0 + location;
                        res_data[offset + location] =  (int32_t) bgr_data[location2]
                                            - (int32_t) bgr_data[location_src + c];
                        location += 1;
                    }
                }
            }
        }
    }
}


int decode_video(
    PyArrayObject ** bgr_arr,
    PyArrayObject ** mv_arr,
    PyArrayObject ** res_arr,
    int representation,
    int accumulate,
    int begin_pos,
    int end_pos) {

    AVCodec *pCodec;
    AVCodecContext *pCodecCtx= NULL;
    AVCodecParserContext *pCodecParserCtx=NULL;

    FILE *fp_in;
    AVFrame *pFrame;
    AVFrame *pFrameBGR;

    const int in_buffer_size=4096;
    uint8_t in_buffer[in_buffer_size + FF_INPUT_BUFFER_PADDING_SIZE];
    memset(in_buffer + in_buffer_size, 0, FF_INPUT_BUFFER_PADDING_SIZE);

    uint8_t *cur_ptr;
    int cur_size;
    int cur_gop = -1;
    AVPacket packet;
    int ret, got_picture;

    avcodec_register_all();

    pCodec = avcodec_find_decoder(AV_CODEC_ID_MPEG4);
    if (!pCodec) {
        printf("Codec not found\n");
        return -1;
    }
    pCodecCtx = avcodec_alloc_context3(pCodec);
    if (!pCodecCtx){
        printf("Could not allocate video codec context\n");
        return -1;
    }
    pCodecParserCtx=av_parser_init(AV_CODEC_ID_MPEG4);
    if (!pCodecParserCtx){
        printf("Could not allocate video parser context\n");
        return -1;
    }
    AVDictionary *opts = NULL;
    av_dict_set(&opts, "flags2", "+export_mvs", 0);
    if (avcodec_open2(pCodecCtx, pCodec, &opts) < 0) {
        printf("Could not open codec\n");
        return -1;
    }
    //Input File
    fp_in = fopen(filename, "rb");
    if (!fp_in) {
        printf("Could not open input stream\n");
        return -1;
    }

    int cur_pos = 0;

    pFrame = av_frame_alloc();
    pFrameBGR = av_frame_alloc();

    uint8_t *buffer;

    av_init_packet(&packet);

    int *accu_src = NULL;
    int *accu_src_old = NULL;

    while (1) {
        cur_size = fread(in_buffer, 1, in_buffer_size, fp_in);
        if (cur_size == 0 || cur_pos > end_pos)
            break;
        cur_ptr=in_buffer;

        while (cur_size>0 && cur_pos<=end_pos){
            int len = av_parser_parse2(
                pCodecParserCtx, pCodecCtx,
                &packet.data, &packet.size,
                cur_ptr , cur_size ,
                AV_NOPTS_VALUE, AV_NOPTS_VALUE, AV_NOPTS_VALUE);
            cur_ptr += len;
            cur_size -= len;

            if(packet.size==0)
                continue;

            ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, &packet);
            if (ret < 0) {
                printf("Decode Error.\n");
                return -1;
            }
            int h = pFrame->height;
            int w = pFrame->width;

            // Initialize arrays.
            if (! (*bgr_arr)) {
                npy_intp dims[4];
                dims[0] = 2;
                dims[1] = h;
                dims[2] = w;
                dims[3] = 3;
                *bgr_arr = PyArray_ZEROS(4, dims, NPY_UINT8, 0);
            }

            if (representation == MV && ! (*mv_arr)) {
                npy_intp dims[4];
                dims[0] = (end_pos - begin_pos + 1);
                dims[1] = h;
                dims[2] = w;
                dims[3] = 2;
                *mv_arr = PyArray_ZEROS(4, dims, NPY_INT32, 0);
            }

            if (representation == RESIDUAL && ! (*res_arr)) {
                npy_intp dims[4];
                dims[0] = (end_pos - begin_pos + 1);
                dims[1] = h;
                dims[2] = w;
                dims[3] = 3;

                *mv_arr = PyArray_ZEROS(4, dims, NPY_INT32, 0);
                *res_arr = PyArray_ZEROS(4, dims, NPY_INT32, 0);
            }

            if ((representation == MV ||
                 representation == RESIDUAL) && accumulate &&
                !accu_src && !accu_src_old) {
                accu_src     = (int*) malloc(w * h * 2 * sizeof(int));
                accu_src_old = (int*) malloc(w * h * 2 * sizeof(int));

                for (size_t x = 0; x < w; ++x) {
                    for (size_t y = 0; y < h; ++y) {
                        accu_src_old[x * h * 2 + y * 2    ]  = x;
                        accu_src_old[x * h * 2 + y * 2 + 1]  = y;
                    }
                }
                memcpy(accu_src, accu_src_old, h * w * 2 * sizeof(int));
            }

            if (pCodecParserCtx->pict_type == AV_PICTURE_TYPE_I) {
                ++cur_gop;
            }

            if (got_picture) {

                if ((cur_pos == 0              && accumulate  && representation == RESIDUAL) ||
                    (!accumulate && representation == RESIDUAL)) {
                        if (cur_pos >= begin_pos){
                            create_and_load_bgr(
                                    pFrame, pFrameBGR, buffer, bgr_arr, cur_pos, begin_pos);
                        }
                }

                if (representation == MV ||
                    representation == RESIDUAL) {
                    AVFrameSideData *sd;
                    sd = av_frame_get_side_data(pFrame, AV_FRAME_DATA_MOTION_VECTORS);
                    if (sd) {
                          if (cur_pos >= begin_pos) {
                                create_and_load_mv_residual(
                                    sd,
                                    *bgr_arr, *mv_arr, *res_arr,
                                    begin_pos,
                                    cur_pos,
                                    accumulate,
                                    representation,
                                    accu_src,
                                    accu_src_old,
                                    w,
                                    h);
                          }
                    }

                }
                cur_pos ++;
            }
        }
    }

    //Flush Decoder
    packet.data = NULL;
    packet.size = 0;
    while(1){
        ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, &packet);
        if (ret < 0) {
            printf("Decode Error.\n");
            return -1;
        }
        if (!got_picture) {
            break;
        } else { // if (cur_gop == gop_target) {
            if ((cur_pos == 0 && accumulate) || !accumulate){
                /*(cur_pos == pos_target - 1 && !accumulate) ||*/
                /*cur_pos == pos_target) {*/
                create_and_load_bgr(
                    pFrame, pFrameBGR, buffer, bgr_arr, cur_pos, begin_pos);
            }
        }
    }

    fclose(fp_in);

    av_parser_close(pCodecParserCtx);

    av_frame_free(&pFrame);
    av_frame_free(&pFrameBGR);
    avcodec_close(pCodecCtx);
    av_free(pCodecCtx);
    if ((representation == MV ||
         representation == RESIDUAL) && accumulate) {
        if (accu_src) {
            free(accu_src);
        }
        if (accu_src_old) {
            free(accu_src_old);
        }
    }

    return 0;
}


void count_frames(int* gop_count, int* frame_count) {

    AVCodec *pCodec;
    AVCodecContext *pCodecCtx= NULL;
    AVCodecParserContext *pCodecParserCtx=NULL;

    FILE *fp_in;

    const int in_buffer_size=4096;
    uint8_t in_buffer[in_buffer_size + FF_INPUT_BUFFER_PADDING_SIZE];
    memset(in_buffer + in_buffer_size, 0, FF_INPUT_BUFFER_PADDING_SIZE);

    uint8_t *cur_ptr;
    int cur_size;
    AVPacket packet;

    avcodec_register_all();

    pCodec = avcodec_find_decoder(AV_CODEC_ID_MPEG4);
    // pCodec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!pCodec) {
        printf("Codec not found\n");
        return -1;
    }
    pCodecCtx = avcodec_alloc_context3(pCodec);
    if (!pCodecCtx){
        printf("Could not allocate video codec context\n");
        return -1;
    }

    pCodecParserCtx=av_parser_init(AV_CODEC_ID_MPEG4);
    // pCodecParserCtx=av_parser_init(AV_CODEC_ID_H264);
    if (!pCodecParserCtx){
        printf("Could not allocate video parser context\n");
        return -1;
    }

    if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0) {
        printf("Could not open codec\n");
        return -1;
    }

    //Input File
    fp_in = fopen(filename, "rb");
    if (!fp_in) {
        printf("Could not open input stream\n");
        return -1;
    }

    *gop_count = 0;
    *frame_count = 0;

    av_init_packet(&packet);

    while (1) {

        cur_size = fread(in_buffer, 1, in_buffer_size, fp_in);
        if (cur_size == 0)
            break;
        cur_ptr=in_buffer;

        while (cur_size>0){

            int len = av_parser_parse2(
                pCodecParserCtx, pCodecCtx,
                &packet.data, &packet.size,
                cur_ptr , cur_size ,
                AV_NOPTS_VALUE, AV_NOPTS_VALUE, AV_NOPTS_VALUE);

            cur_ptr += len;
            cur_size -= len;

            if(packet.size==0)
                continue;
            if (pCodecParserCtx->pict_type == AV_PICTURE_TYPE_I) {
                ++(*gop_count);
            }
            ++(*frame_count);
        }
    }

    fclose(fp_in);
    av_parser_close(pCodecParserCtx);

    avcodec_close(pCodecCtx);
    av_free(pCodecCtx);

    return 0;
}


static PyObject *get_num_gops(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "s", &filename)) return NULL;

    int gop_count, frame_count;
    count_frames(&gop_count, &frame_count);
    return Py_BuildValue("i", gop_count);
}


static PyObject *get_num_frames(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "s", &filename)) return NULL;

    int gop_count, frame_count;
    count_frames(&gop_count, &frame_count);
    return Py_BuildValue("i", frame_count);
}


static PyObject *load(PyObject *self, PyObject *args)
{

    PyObject *arg1 = NULL; // filename.
    int representation, accumulate;
    int begin_pos, end_pos;

    if (!PyArg_ParseTuple(args, "siiii", &filename,
        &representation, &accumulate,
        &begin_pos, &end_pos)) {
        printf("input args error!!\n");
        return NULL;
    }

    PyArrayObject *bgr_arr = NULL;
    PyArrayObject *final_bgr_arr = NULL;
    PyArrayObject *mv_arr = NULL;
    PyArrayObject *res_arr = NULL;

    if(decode_video(&bgr_arr, &mv_arr, &res_arr,
                    representation,
                    accumulate, begin_pos, end_pos) < 0) {
        printf("Decoding video failed.\n");
        Py_XDECREF(bgr_arr);
        Py_XDECREF(mv_arr);
        Py_XDECREF(res_arr);
        return Py_None;
    }
    if(representation == MV) {
        Py_XDECREF(bgr_arr);
        Py_XDECREF(res_arr);
        return mv_arr;

    } else if(representation == RESIDUAL) {
        Py_XDECREF(bgr_arr);
        Py_XDECREF(mv_arr);
        return res_arr;

    } else {
        Py_XDECREF(mv_arr);
        Py_XDECREF(res_arr);

        npy_intp *dims_bgr = PyArray_SHAPE(bgr_arr);
        int h = dims_bgr[1];
        int w = dims_bgr[2];

        npy_intp dims[3];
        dims[0] = h;
        dims[1] = w;
        dims[2] = 3;
        PyArrayObject *final_bgr_arr = PyArray_ZEROS(3, dims, NPY_UINT8, 0);

        int size = h * w * 3 * sizeof(uint8_t);
        memcpy(final_bgr_arr->data, bgr_arr->data + size, size);

        Py_XDECREF(bgr_arr);
        return final_bgr_arr;
    }
}


static PyMethodDef CoviarMethods[] = {
    {"load",  load, METH_VARARGS, "Load a frame."},
    {"get_num_gops",  get_num_gops, METH_VARARGS, "Getting number of GOPs in a video."},
    {"get_num_frames",  get_num_frames, METH_VARARGS, "Getting number of frames in a video."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initcoviar(void)
{
    PyObject *m;
    /*m = PyModule_Create(&coviarmodule);*/
    m = Py_InitModule("coviar", CoviarMethods);
    if (m == NULL)
        return NULL;

    /* IMPORTANT: this must be called */
    import_array();

    CoviarError = PyErr_NewException("coviar.error", NULL, NULL);
    Py_INCREF(CoviarError);
    PyModule_AddObject(m, "error", CoviarError);
    return m;
}


