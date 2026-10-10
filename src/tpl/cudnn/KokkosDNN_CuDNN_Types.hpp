#ifndef KOKKOSDNN_CUDNN_TYPES_HPP
#define KOKKOSDNN_CUDNN_TYPES_HPP

namespace KokkosDNN {
namespace Impl {

enum class CudnnSdpaScalar : int { Float = 0, BFloat16 = 1 };

}  // namespace Impl
}  // namespace KokkosDNN

#endif
