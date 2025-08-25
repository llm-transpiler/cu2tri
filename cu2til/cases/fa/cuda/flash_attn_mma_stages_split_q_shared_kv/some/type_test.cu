#include <cuda_fp16.h>
#include <iostream>
#include <typeinfo>
#include <cstdint>

int main() {
    uint32_t base_ptr = 100;
    int i = 4;
    
    auto result1 = base_ptr + i * 2;
    auto result2 = base_ptr + i * sizeof(half);
    
    std::cout << "i * 2 type: " << typeid(i * 2).name() << std::endl;
    std::cout << "i * sizeof(half) type: " << typeid(i * sizeof(half)).name() << std::endl;
    std::cout << "result1 type: " << typeid(result1).name() << std::endl;  
    std::cout << "result2 type: " << typeid(result2).name() << std::endl;
    
    std::cout << "sizeof(half) = " << sizeof(half) << std::endl;
    std::cout << "sizeof(size_t) = " << sizeof(size_t) << std::endl;
    
    return 0;
}
/*
i * 2 type: i                    // int类型
i * sizeof(half) type: m         // size_t类型
result1 type: j                  // uint32_t类型
result2 type: m                  // size_t类型
sizeof(half) = 2                 // half类型占2字节
sizeof(size_t) = 8               // size_t类型占8字节
*/