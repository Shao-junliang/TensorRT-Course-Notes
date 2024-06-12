#include "cuda-runtime-api.h"

__device__ void affine_project(float *matrix, float x, float y, float *ox, float *oy) {
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

__global__ void decode_kernel(float *predict, int num_bboxes, int num_classes, float confidence_threshold,
                              float *invert_affine_matrix, float *parray, int max_objects, int NUM_BOX_ELEMENT) {
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) { return; }

    // pitem 是指向每一行结果的首地址
    float *pitem = predict + (5 + num_classes) * position;
    // 置信度，表示该预测框中是否有对象的置信度。
    float objectness = pitem[4];
    if (objectness < confidence_threshold) { return; }

    // 当前行的 label 指针，指向类别概率的起始位置。
    float *class_confidence = pitem + 5;
    // 获取 class_confidence 所指向的浮点数的值，将其赋值给 confidence，然后将 class_confidence 指针递增到下一个位置。
    float confidence = *class_confidence++; // *ptr 解引用指针;
    int label = 0;

    // 此处的 for 循环相当于 std::max_element 的作用；
    for (int i = 1; i < num_classes; ++i, ++class_confidence) {
        if (*class_confidence > confidence) {
            confidence = *class_confidence;
            label = i;
        }
    }
    confidence *= objectness;
    if (objectness < confidence_threshold) { return; }

    // 使用原子操作 atomicAdd 递增并获取当前输出索引 output_index。
    /*
     * parray = [count, box1, box2, ……]
     * atomicAdd(parray, 1) 的操作相当于是 count += 1, 但是返回的是没有 +1 之前的 old_count;
     */
    int index = atomicAdd(parray, 1);
    if (index >= max_objects) { return; }

    // 获取 left, top, width, height 的值
    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;

    // xywh to xyxy
    float left = cx - width * 0.5;
    float top = cy - height * 0.5;
    float right = cx + width * 0.5;
    float bottom = cy + height * 0.5;

    /*
     * 仿射变换
     * affine_project(invert_affine_matrix, left,  top,    &left,  &top);
     * affine_project(invert_affine_matrix, right, bottom, &right, &bottom);
     */

    // 将转换好的数据进行存储；
    float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; // 用于nms的标志位，1 = keep, 0 = ignore
}

__device__ float box_iou(float aleft, float atop, float aright, float abottom,
                         float bleft, float btop, float bright, float bbottom) {
    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);

    if (c_area == 0.0f) { return 0.0f; }
    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

__global__ void fast_nms_kernel(float *bboxes, int max_objects, float threshold, int NUM_BOX_ELEMENT) {
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    // (int)*bboxes 解引用获取第一个计数器的值 count;
    int count = min((int)*bboxes, max_objects);
    if (position >= count) { return; }

    // left, top, right, bottom, confidence, class, keepflag
    float *pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for (int i = 0; i < count; ++i) {
        float *pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if (i == position || pcurrent[5] != pitem[5]) { continue; }

        if (pitem[4] >= pcurrent[4]) {
            // 过滤
            if (pitem[4] == pcurrent[4] && i < position) { continue; }

            float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                                pitem[0], pitem[1], pitem[2], pitem[3]);

            if (iou > threshold) {
                pcurrent[6] = 0;
                return;
            }
        }
    }
}

void decode_kernel_invoker(
    float *predict, int num_bboxes, int num_classes, float confidence_threshold,
    float nms_threshold, float *invert_affine_matrix, float *parray, int max_objects,
    int NUM_BOX_ELEMENT, cudaStream_t stream) {
    auto block = num_bboxes > 512 ? 512 : num_bboxes;
    auto grid = (num_bboxes + block - 1) / block;

    // [left, top, width, height, confidence,label1, label2,……] ---> [left, top, width, height, confidence,label,nms_symbol]
    decode_kernel<<<grid, block, 0, stream>>>(
        predict, num_bboxes, num_classes, confidence_threshold,
        invert_affine_matrix, parray, max_objects, NUM_BOX_ELEMENT);

    block = max_objects > 512 ? 512 : max_objects;
    grid = (max_objects + block - 1) / block;
    fast_nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold, NUM_BOX_ELEMENT);
}