import torch

def convert_torch_model(source, target):
    """
    you should run this code with the high version torch
    """
    source_model = torch.load(source)

    torch.save(source_model, target, _use_new_zipfile_serialization=False)

if __name__ == '__main__':

    source = "target_DOTA_ten_class_eta_0.1_local_True_global_True_gamma_5_session_1_epoch_20_step_10000.pth"
    target = "tmp_epoch_10.pth"

    convert_torch_model(source, target)