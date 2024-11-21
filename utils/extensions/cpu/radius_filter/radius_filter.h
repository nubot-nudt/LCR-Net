#pragma once

#include <torch/torch.h>

std::vector<std::vector<torch::Tensor>> radius_filter(torch::Tensor nodes_dict, torch::Tensor length_dict, float radius);
