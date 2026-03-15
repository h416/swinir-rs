#include <torch/torch.h>
#include <stdexcept>

using namespace std;

extern thread_local char *torch_last_err;

static at::Device device_of_int(int d) {
    if (d == -3) return at::Device(at::kVulkan);
    if (d == -2) return at::Device(at::kMPS);
    if (d < 0) return at::Device(at::kCPU);
    return at::Device(at::kCUDA, /*index=*/d);
}

extern "C" {

void atm_to_device(torch::jit::script::Module *m, int device) {
    try {
        m->to(device_of_int(device));
    } catch (const exception& e) {
        torch_last_err = strdup(e.what());
    }
}

// Convert only floating-point parameters/buffers to the target dtype,
// leaving integer tensors (e.g. index buffers) untouched.
static void convert_float_attrs(torch::jit::script::Module mod, at::ScalarType target) {
    for (const auto& attr : mod.named_attributes(/*recurse=*/false)) {
        if (attr.value.isTensor()) {
            auto t = attr.value.toTensor();
            if (t.is_floating_point()) {
                mod.setattr(attr.name, t.to(target));
            }
        }
    }
    for (const auto& child : mod.named_children()) {
        convert_float_attrs(child.value, target);
    }
}

void atm_to_dtype(torch::jit::script::Module *m, int dtype) {
    try {
        auto scalar_type = static_cast<at::ScalarType>(dtype);
        convert_float_attrs(*m, scalar_type);
    } catch (const exception& e) {
        torch_last_err = strdup(e.what());
    }
}

}
