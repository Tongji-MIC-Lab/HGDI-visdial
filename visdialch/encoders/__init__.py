from visdialch.encoders.HGDI import HGDI

def Encoder(model_config, *args):
    name_enc_map = {
    	"hgdi": HGDI
    }
    return name_enc_map[model_config["encoder"]](model_config, *args)
