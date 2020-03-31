import matplotlib.pyplot as plt
win_size = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# ber_c_d = [0.20114587301587306, 0.19699103174603175,  0.1940080555555556, 0.19255412698412697, 0.19421003968253966,
# #            0.19267087301587302, 0.19228896825396827, 0.19111960317460314, 0.19113968253968258, 0.19148488095238095,
# #            0.19083726190476194]
# #
# # ber_n_d = [0.23349583333333335, 0.2150283333333333,  0.2043316666666667, 0.18693249999999997,  0.18459250000000002,
# #            0.18588666666666667, 0.18692166666666665, 0.2077625, 0.2110816666666667, 0.21380333333333335,
# #            0.21020333333333333]
# # plt.plot(win_size, ber_c_d, 'b', win_size, ber_n_d , 'r')
# # plt.legend(['classic_mod + tree_demod', 'neural_mod + tree_demod'])
# # plt.xlabel('sliding window size')
# # plt.ylabel('round-trip ber')
# # plt.title('SNR=2.2db, hidden_layer=20, max_depth=5, proj_num=50, num_iter=800')
# # plt.show()

ber_n_d_random = [0.0270475, 0.0290975, 0.022656667, 0.0257975, 0.0259725, 0.024585833,
0.021246667, 0.0220725, 0.021951667, 0.021976667, 0.02117, 0.02354, 0.0208275, 0.02074, 0.020439167,
0.0216525,0.027865, 0.020670833]

ber_n_d_fixed = [0.028836667, 0.024513333, 0.023085833, 0.023190833, 0.02322, 0.021701667,
0.019206667, 0.025898333, 0.02282, 0.02297, 0.020881667, 0.022106667, 0.0207625, 0.0204475, 0.019103333,
0.021668333, 0.019963333, 0.020235833]
plt.plot(win_size, ber_n_d_random, 'b', win_size, ber_n_d_fixed , 'r')
plt.legend(['neural_mod + tree_demod + random forgetting', 'neural_mod + tree_demod + FIFO'])
plt.xlabel('sliding window size')
plt.ylabel('round-trip ber')
plt.title('SNR=4.2db, hidden_layer=20, max_depth=5, proj_num=50, num_iter=800')
plt.show()

ber_c_d_random = [0.023095833, 0.02171, 0.022000833, 0.0226275, 0.022376667,
                  0.022851667, 0.02544, 0.022955, 0.022063333, 0.021713333, 0.021629167,
                  0.023279167, 0.023443333, 0.020444167, 0.022030833, 0.0200775, 0.02018,
                  0.020504167]

ber_c_d_fixed = [0.0260425, 0.023680833, 0.024336667, 0.021661667, 0.026523333,
                 0.0227175, 0.0250925, 0.02408, 0.024295833, 0.024348333, 0.024348333, 0.025008333,
                 0.024575, 0.024831667, 0.024675833, 0.024675833, 0.024675833, 0.019828333]
plt.plot(win_size, ber_c_d_random, 'b', win_size, ber_c_d_fixed , 'r')
plt.legend(['class_mod + tree_demod + random forgetting', 'class_mod + tree_demod + FIFO'])
plt.xlabel('sliding window size')
plt.ylabel('round-trip ber')
plt.title('SNR=4.2db, hidden_layer=20, max_depth=5, proj_num=50, num_iter=800')
plt.show()