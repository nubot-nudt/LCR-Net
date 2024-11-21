import torch
import argparse


def load_snapshot(snapshot, snapshot_global_description_head):
    state_dict1 = torch.load(snapshot, map_location=torch.device('cpu'), weights_only=True)
    assert 'model' in state_dict1, 'No model can be loaded from {snapshot}.'

    state_dict2 = torch.load(snapshot_global_description_head, map_location=torch.device('cpu'), weights_only=True)
    assert 'model' in state_dict2, 'No model can be loaded2 from {snapshot_global_description_head}.'

    model_dict_reg = state_dict1['model']
    model_dict_ld = state_dict2['model']
    
    for key in model_dict_ld.keys():
        if 'netvlad' in key:
            model_dict_reg[key] = model_dict_ld[key]

    state_dict1['model'] = model_dict_reg

    torch.save(state_dict1, 'merged_model.tar')
    print('Merged model has been saved as merged_model.pth')


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_reg', type=str, help='')
    parser.add_argument('--model_ld', type=str, help='')
    return parser


def main():
    args = parser().parse_args()
    load_snapshot(args.model_reg, args.model_ld)


if __name__ == '__main__':
    main()