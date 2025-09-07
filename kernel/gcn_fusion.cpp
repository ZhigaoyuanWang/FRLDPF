#include <torch/extension.h>

std::vector<torch::Tensor> gcn_fusion_forward(
    torch::Tensor embeddings_Q,
    torch::Tensor embeddings_K,
    torch::Tensor input_x) {
    
    torch::Tensor GCN_Q_pre_softmax = torch::mm(embeddings_Q, embeddings_Q.t());
    torch::Tensor GCN_K_pre_softmax = torch::mm(embeddings_K, embeddings_K.t());

    torch::Tensor GCN_Q = torch::softmax(GCN_Q_pre_softmax, 0);
    torch::Tensor GCN_K = torch::softmax(GCN_K_pre_softmax, 0);

    torch::Tensor x2 = input_x.reshape({-1, input_x.size(-1)});

    torch::Tensor Q = torch::mm(x2, GCN_Q);
    torch::Tensor K = torch::mm(x2, GCN_K);

    return {Q, K, GCN_Q, GCN_K, input_x, embeddings_Q, embeddings_K};
}


std::vector<torch::Tensor> gcn_fusion_backward(
    torch::Tensor grad_Q_output,
    torch::Tensor grad_K_output,
    torch::Tensor GCN_Q,
    torch::Tensor GCN_K,
    torch::Tensor input_x,
    torch::Tensor embeddings_Q,
    torch::Tensor embeddings_K) {

    
    torch::Tensor x2 = input_x.reshape({-1, input_x.size(-1)});

    torch::Tensor grad_x2_Q = torch::mm(grad_Q_output, GCN_Q.t());
    torch::Tensor grad_x2_K = torch::mm(grad_K_output, GCN_K.t());

    torch::Tensor grad_GCN_Q_softmax = torch::mm(x2.t(), grad_Q_output);
    torch::Tensor grad_GCN_K_softmax = torch::mm(x2.t(), grad_K_output);

    torch::Tensor grad_GCN_Q = grad_GCN_Q_softmax * GCN_Q - (grad_GCN_Q_softmax * GCN_Q).sum(0, true) * GCN_Q;
    torch::Tensor grad_GCN_K = grad_GCN_K_softmax * GCN_K - (grad_GCN_K_softmax * GCN_K).sum(0, true) * GCN_K;


    torch::Tensor grad_embeddings_Q_mm = torch::mm(grad_GCN_Q, embeddings_Q) + torch::mm(grad_GCN_Q.t(), embeddings_Q);
    torch::Tensor grad_embeddings_K_mm = torch::mm(grad_GCN_K, embeddings_K) + torch::mm(grad_GCN_K.t(), embeddings_K);


    torch::Tensor grad_input_x = (grad_x2_Q + grad_x2_K).reshape(input_x.sizes());
    torch::Tensor grad_embeddings_Q = grad_embeddings_Q_mm;
    torch::Tensor grad_embeddings_K = grad_embeddings_K_mm;

    return {grad_embeddings_Q, grad_embeddings_K, grad_input_x};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gcn_fusion_forward", &gcn_fusion_forward, "GCN Fusion Forward (C++)");
    m.def("gcn_fusion_backward", &gcn_fusion_backward, "GCN Fusion Backward (C++)");
}