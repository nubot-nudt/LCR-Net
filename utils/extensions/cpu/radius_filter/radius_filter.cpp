#include "radius_filter.h"

std::vector<std::vector<torch::Tensor>> radius_filter(torch::Tensor nodes_dict, torch::Tensor length_dict, float radius) {

    int length_increment = 0;
    std::vector<torch::Tensor> masks;
    std::vector<torch::Tensor> nms_length;

    for (int i = 0; i < length_dict.size(0); i++) {
        int length = length_increment + length_dict[i].item<int>();
        torch::Tensor nodes = nodes_dict.slice(0, length_increment, length);
        length_increment = length;

        int valid_node_counter = 0;
        torch::Tensor mask = torch::zeros({nodes.size(0)}, torch::kBool).cuda();
        mask[0] = true;


        // torch::Tensor neighbour = -torch::ones({nodes.size(0), 32}, torch::long).cuda();
        // torch::Tensor num = torch::ones({nodes.size(0)}, torch::long).cuda();
        // std::vector<torch::long> indx = {0};

        for (int idx = 1; idx < nodes.size(0); idx++) {
            torch::Tensor dis = torch::sqrt(torch::sum(torch::pow(nodes[idx].unsqueeze(0) - nodes.masked_select(mask.unsqueeze(1)).view({valid_node_counter+1, -1}), 2), 1));

            if ((dis > radius).sum().item<int>() == valid_node_counter+1) {
                mask[idx] = true;
                valid_node_counter++;
            }
        }

        masks.push_back(mask);
        nms_length.push_back((mask.sum()));
    }

    return {masks, nms_length};
}
