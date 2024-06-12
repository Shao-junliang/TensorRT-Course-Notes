#include "cuda-runtime-api.h"

__host__ __device__ int sort_func(int a, int b) {
    return a > b;
}

void thrust_demo() {
    int data[] = {5, 3, 1, 5, 2, 0};
    int ndata = sizeof(data) / sizeof(data[0]);
    thrust::host_vector<int> array1(data, data + ndata);
    thrust::sort(array1.begin(), array1.end(), sort_func);

    // thrust::device_vector<int> array2 = thrust::host_vector<int>(data, data + ndata);
    // thrust::sort(array2.begin(), array2::end(), [] __device__(int a, int b) { return a < b; });

    printf("array1------------------------\n");
    for (int i = 0; i < array1.size(); ++i)
        std::cout << array1[i] << std::endl;

    // printf("array2------------------------\n");
    // for (int i = 0; i < array2.size(); ++i)
    //     std::cout << array2[i] << std::endl;
    return;
}