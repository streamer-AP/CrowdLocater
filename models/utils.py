

def module2model(module_state_dict):
    state_dict = {}
    for k, v in module_state_dict.items():
        while k.startswith("module."):
            k = k[7:]
        #while apply ema model
        if k=="n_averaged":
            print(f"{k}:{v}")
            continue
        state_dict[k] = v
    return state_dict