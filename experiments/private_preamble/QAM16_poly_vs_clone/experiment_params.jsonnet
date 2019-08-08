local params = import '../model_params.libsonnet';
local bps = 4;
local opt = 'adam';
local signal_power = 1.0;
{
    "num_trials": 50,
    "train_SNR_dbs": [
        20.0,
        15.0,
        10.4
    ],
    "base": {
        "__meta__": {
            "protocol": "private_preamble",
            "experiment_name": "QAM16_poly_vs_clone",
            "mod_order": "QAM16",
            "random_seed": "placeholder",
            "numpy_seed": "placeholder",
            "torch_seed": "placeholder",
            "verbose": false
        },
        "test_batch_size": 100000,
        "test_SNR_db_type": "ber_roundtrip",
        "bits_per_symbol": bps,
        "batch_size": 128,
        "num_iterations": 20000,
        "results_every": 1000,
        "signal_power": signal_power,
        "agent1": {
            "bits_per_symbol": bps,
            "max_amplitude": signal_power,
            "optimizer": opt,
            "mod_model": "poly",
            "mod_params": params['poly_mod_qam16'],
            "demod_model": "poly",
            "demod_params": params['poly_demod_qam16']
        },
        "agent2": {
            "bits_per_symbol": bps,
            "max_amplitude": signal_power,
            "optimizer": opt,
            "mod_model": "clone",
            "demod_model": "clone"
        }
    }
}