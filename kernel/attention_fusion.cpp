#include <torch/extension.h>
#include <ATen/ATen.h>

std::vector<torch::Tensor> attention_fusion_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor inter_embeddings,
    torch::Tensor d,
    torch::Tensor x) {
    
    Q = Q + inter_embeddings;
    K = K + inter_embeddings;

    torch::Tensor attention_score = torch::softmax(torch::mm(Q, K.t()) / d.sqrt(), 1);

    torch::Tensor output = torch::einsum("ij,jkl->ikl", {attention_score, V}) + x;

    return {output, attention_score, Q, K, V};
}

std::vector<torch::Tensor> attention_fusion_backward(
    torch::Tensor grad_output,
    torch::Tensor attention_score,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor x) {
    
    torch::Tensor grad_V = torch::einsum("ij,jkl->jkl", {attention_score.t(), grad_output});
    torch::Tensor grad_attention_score = torch::einsum("jkl,ikl->ij", {V, grad_output});

    torch::Tensor grad_softmax = torch::softmax_backward_data(grad_attention_score, attention_score, 1);
    torch::Tensor grad_Q = torch::mm(grad_softmax, K) / Q.sizes()[0];
    torch::Tensor grad_K = torch::mm(grad_softmax.t(), Q) / Q.sizes()[0];
    
    torch::Tensor grad_inter = grad_Q + grad_K;

    torch::Tensor grad_x = grad_output + grad_V;

    return {grad_Q, grad_K, grad_V, grad_inter, grad_x};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attention_fusion_forward, "Attention Fusion forward pass");
    m.def("backward", &attention_fusion_backward, "Attention Fusion backward pass");
}