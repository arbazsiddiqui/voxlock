import wesep.models.tse_bsrnn_spk as bsrnn_spk

def get_model(model_name: str):
    if model_name.startswith("TSE_BSRNN_SPK"):
        return getattr(bsrnn_spk, model_name)
    else:  # model_name error !!!
        print(model_name + " not found !!!")
        exit(1)
