#include "cuda-runtime-api.h"

#define min(a, b) ((a) < (b) ? (a) : (b))

// typedef <existing_type> <new_name>; 用于给已存在的类型起别名
typedef unsigned char uint8_t;

/*
 * 旋转变换：
 * [x'] = [ cos(delta), sin(delta)] [x]
 * [y'] = [-sin(delta), cos(delta)] [y]
 * 缩放变换：
 * [x'] = [scale, 0] [x]
 * [y'] = [0, scale] [y]
 * 平移变换：
 * [x'] = [1, 0] [x] + [ox]
 * [y'] = [0, 1] [y] + [oy]
 * 旋转 + 缩放：
 * [x'] = [ cos(delta)*scale, sin(delta)*scale] [x]
 * [y'] = [-sin(delta)*scale, cos(delta)*scale] [y]
 * 旋转 + 缩放 + 平移：
 * [x'] = [ cos(delta)*scale, sin(delta)*scale, ox] [x]
 * [y'] = [-sin(delta)*scale, cos(delta)*scale, oy] [y]
 * [w'] = [        0        ,        0        ,  1] [1]
 */

struct AffineMatrix {
    float i2d[6];
    float d2i[6];

    /*
     * 求解imat的逆矩阵，仿射变换推导过程中第三行为[0, 0, 1]，所以此处简写为 3x2 的矩阵；
     */
    void invertAffineTransfrom(float imat[6], float omat[6]) {
        float i00 = imat[0];
        float i01 = imat[1];
        float i02 = imat[2];
        float i10 = imat[3];
        float i11 = imat[4];
        float i12 = imat[5];

        // 行列式计算，用于判断矩阵是否可逆，如果行列式为0，该矩阵不可逆。否则，计算其倒数
        float D = i00 * i11 - i01 * i10;
        D = D != 0 ? 1.0 / D : D;

        // 计算剩余的伴随矩阵除以行列式
        float A11 = i11 * D;
        float A22 = i00 * D;
        float A12 = -i01 * D;
        float A21 = -i10 * D;

        // 计算新的平移分量 b1 和 b2：
        float b1 = -A11 * i02 - A12 * i12;
        float b2 = -A21 * i02 - A22 * i12;

        // 结果输出
        omat[0] = A11;
        omat[1] = A12;
        omat[2] = b1;
        omat[3] = A21;
        omat[4] = A22;
        omat[5] = b2;
    }

    /*
     * 该函数包含：缩放变换，两次平移变换。
     * 缩放是将输入的from图像，等比缩放scale倍，缩放到到to尺度下
     * 第一次平移是将图像的原点，从左上角，移动到缩放(scale)后图像的中心上
     * 第二次平移是将图像从原点移动到目标（to）图的中心上
     * [x'] = [scale,   0  , -scale * from.width  * 0.5 + to.width  * 0.5] [x]
     * [y'] = [  0  , scale, -scale * from.height * 0.5 + to.height * 0.5] [y]
     * [z'] = [  0  ,   0  ,                      1                      ] [1]
     * '+ scale * 0.5 - 0.5' 的原因是使得中心更加对齐，下采样不明显，但是上采样时会比较明显；
     */
    void compute(const cv::Size &from, const cv::Size &to) {
        // 宽与高 放大或者缩小的比例系数
        float scale_x = to.width / (float)from.width;
        float scale_y = to.height / (float)from.height;

        float scale = min(scale_x, scale_y);

        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] = -scale * from.width * 0.5 + to.width * 0.5 + scale * 0.5 - 0.5;
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;
        invertAffineTransfrom(i2d, d2i);
    }
};

__device__ void affine_project(float *matrix, int x, int y, float *proj_x, float *proj_y) {
    /*
     * 在 CUDA 中，内核函数参数是通过指针传递的，因为内核函数在设备上执行，而数据可能位于设备内存中。
     * 使用指针和解引用可以让内核函数直接操作设备内存中的数据，从而实现计算结果的存储和传递。
     * 在本例中，*proj_x 和 *proj_y 是对指针进行解引用，以便将计算结果存储到这些指针指向的地址处。
     */
    *proj_x = matrix[0] * x + matrix[1] * y + matrix[2];
    *proj_y = matrix[3] * x + matrix[4] * y + matrix[5];
}

__global__ void warp_affine_bilinear_kernel(
    uint8_t *src, int src_line_size, int src_width, int src_height,
    uint8_t *dst, int dst_line_size, int dst_width, int dst_height,
    uint8_t fill_value, AffineMatrix matrix) {
    // 计算索引，相当于图像的坐标（x，y）
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;

    // 角标越界
    if (dx >= dst_width || dy >= dst_height) { return; }

    float c0 = fill_value, c1 = fill_value, c2 = fill_value;
    float src_x = 0;
    float src_y = 0;

    // 计算仿射变换后的坐标，通过仿射变换矩阵计算出这个目标位置对应的源图像中的位置;
    affine_project(matrix.d2i, dx, dy, &src_x, &src_y);

    if (src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height) {
        // 如果超出了源图像的范围，则填充为 fill_value
    } else {
        // 如果在源图像范围内，进行双线性插值计算

        /*
         * 使用 floorf 函数获取插值区域的左上角顶点坐标
         * floorf 函数用于计算小于或等于给定浮点数的最大整数，并返回该整数值。
         * 左上角整数坐标 (x_low, y_low)，并确定右下角坐标 (x_high, y_high)。
         */
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        // 定义一个数组来存储插值区域的四个像素的值
        uint8_t const_values[] = {fill_value, fill_value, fill_value};

        // ly 和 lx：计算 src_y 和 src_x 的小数部分，用于确定权重。
        float ly = src_y - y_low;
        float lx = src_x - x_low;

        // hy 和 hx：是 ly 和 lx 对应的余数部分。
        float hy = 1 - ly;
        float hx = 1 - lx;

        // w1, w2, w3, w4：分别是四个插值点的权重，计算方法是余数部分的乘积。
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

        // 初始化四个指针 v1, v2, v3, v4，默认指向 const_values，即默认填充值。
        uint8_t *v1 = const_values;
        uint8_t *v2 = const_values;
        uint8_t *v3 = const_values;
        uint8_t *v4 = const_values;

        // 根据插值区域的位置确定四个像素的位置和值
        if (y_low >= 0) {
            if (x_low >= 0) {
                // 计算源图像中 (x_low, y_low) 处像素的地址。* 3 是因为每个像素有三个通道（RGB）。
                v1 = src + y_low * src_line_size + x_low * 3;
            }
            if (x_high < src_width) {
                // 计算源图像中 (x_high, y_low) 处像素的地址。
                v2 = src + y_low * src_line_size + x_high * 3;
            }
        }
        if (y_high < src_height) {
            if (x_low >= 0) {
                // 计算源图像中 (x_low, y_high) 处像素的地址。
                v3 = src + y_high * src_line_size + x_low * 3;
            }
            if (x_high < src_width) {
                // 计算源图像中 (x_high, y_high) 处像素的地址。
                v4 = src + y_high * src_line_size + x_high * 3;
            }
        }

        // 使用权重 w1, w2, w3, w4 对四个像素值进行加权平均，计算出目标图像中当前像素的值。加0.5用于四舍五入， floorf 用于取整。
        c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
        c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
        c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
    }
    // 计算目标图像中当前像素的地址 pdst。
    uint8_t *pdst = dst + dy * dst_line_size + dx * 3;
    pdst[0] = c0;
    pdst[1] = c1;
    pdst[2] = c2;
}

void warp_affine_bilinear(
    uint8_t *src, int src_line_size, int src_width, int src_height,
    uint8_t *dst, int dst_line_size, int dst_width, int dst_height,
    uint8_t fill_value) {
    // 此处的grids，blocks，是将 dst_img 按照 32x32 的小块进行分割
    dim3 block_size(32, 32);
    dim3 grid_size((dst_width + 31) / 32, (dst_height + 31) / 32);

    AffineMatrix affine;
    affine.compute(cv::Size(src_width, src_height), cv::Size(dst_width, dst_height));

    warp_affine_bilinear_kernel<<<grid_size, block_size, 0, nullptr>>>(
        src, src_line_size, src_width, src_height,
        dst, dst_line_size, dst_width, dst_height,
        fill_value, affine);
}