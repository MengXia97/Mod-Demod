from utils.util_lookup_table import BER_lookup_table
tbl = BER_lookup_table()

snr = tbl.get_optimal_SNR_for_BER(0.02, bits_per_symbol=2)
print(snr)