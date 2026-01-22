import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

import photoelastimetry.io

_virino_list = [
    [9.8857443e-01, 9.9435500e-01, 6.2314191e-01],
    [9.8899847e-01, 9.8635272e-01, 5.7966316e-01],
    [9.8942251e-01, 9.7835044e-01, 5.3618441e-01],
    [9.8984654e-01, 9.7034816e-01, 4.9270565e-01],
    [9.9027058e-01, 9.6234589e-01, 4.4922690e-01],
    [9.9069462e-01, 9.5434361e-01, 4.0574814e-01],
    [9.9111865e-01, 9.4634133e-01, 3.6226939e-01],
    [9.9154269e-01, 9.3833905e-01, 3.1879064e-01],
    [9.9196673e-01, 9.3033677e-01, 2.7531188e-01],
    [9.9239076e-01, 9.2233449e-01, 2.3183313e-01],
    [9.9281480e-01, 9.1433222e-01, 1.8835437e-01],
    [9.9323884e-01, 9.0632994e-01, 1.4487562e-01],
    [9.7486175e-01, 9.0365009e-01, 1.3052945e-01],
    [9.5578911e-01, 9.0112913e-01, 1.1842435e-01],
    [9.3643831e-01, 8.9863835e-01, 1.0838942e-01],
    [9.1682224e-01, 8.9616388e-01, 1.0091429e-01],
    [8.9694689e-01, 8.9369374e-01, 9.6447314e-02],
    [8.7684153e-01, 8.9120827e-01, 9.5258267e-02],
    [8.5653021e-01, 8.8869044e-01, 9.7346188e-02],
    [8.3603678e-01, 8.8612534e-01, 1.0242664e-01],
    [8.1538886e-01, 8.8349708e-01, 1.1002308e-01],
    [7.9461828e-01, 8.8079066e-01, 1.1958994e-01],
    [7.7375469e-01, 8.7799025e-01, 1.3061689e-01],
    [7.5283007e-01, 8.7508386e-01, 1.4267203e-01],
    [7.3187714e-01, 8.7206028e-01, 1.5542025e-01],
    [7.1092673e-01, 8.6890776e-01, 1.6860465e-01],
    [6.9001163e-01, 8.6561817e-01, 1.8203636e-01],
    [6.6915888e-01, 8.6218355e-01, 1.9557406e-01],
    [6.4839731e-01, 8.5859973e-01, 2.0911664e-01],
    [6.2775299e-01, 8.5486074e-01, 2.2258729e-01],
    [6.0725065e-01, 8.5096547e-01, 2.3592847e-01],
    [5.8691326e-01, 8.4691095e-01, 2.4909854e-01],
    [5.6676106e-01, 8.4269805e-01, 2.6206608e-01],
    [5.4681294e-01, 8.3832565e-01, 2.7480488e-01],
    [5.2708867e-01, 8.3379687e-01, 2.8729780e-01],
    [5.0760525e-01, 8.2911157e-01, 2.9952708e-01],
    [4.8837974e-01, 8.2427446e-01, 3.1148185e-01],
    [4.6942432e-01, 8.1928708e-01, 3.2315600e-01],
    [4.5075478e-01, 8.1415471e-01, 3.3454100e-01],
    [4.3238302e-01, 8.0888059e-01, 3.4563345e-01],
    [4.1432398e-01, 8.0346971e-01, 3.5642837e-01],
    [3.9659324e-01, 7.9792520e-01, 3.6692012e-01],
    [3.7920436e-01, 7.9225255e-01, 3.7710674e-01],
    [3.6216978e-01, 7.8645707e-01, 3.8698773e-01],
    [3.4550374e-01, 7.8054354e-01, 3.9656225e-01],
    [3.2922347e-01, 7.7451612e-01, 4.0582982e-01],
    [3.1334721e-01, 7.6837977e-01, 4.1478614e-01],
    [2.9789250e-01, 7.6214030e-01, 4.2343414e-01],
    [2.8287822e-01, 7.5580366e-01, 4.3177534e-01],
    [2.6832607e-01, 7.4937301e-01, 4.3980975e-01],
    [2.5426331e-01, 7.4285433e-01, 4.4753557e-01],
    [2.4071320e-01, 7.3625240e-01, 4.5495520e-01],
    [2.2770577e-01, 7.2957297e-01, 4.6207283e-01],
    [2.1527283e-01, 7.2281963e-01, 4.6888923e-01],
    [2.0344956e-01, 7.1599814e-01, 4.7540402e-01],
    [1.9227218e-01, 7.0911249e-01, 4.8162335e-01],
    [1.8177959e-01, 7.0216821e-01, 4.8754862e-01],
    [1.7201178e-01, 6.9516993e-01, 4.9318382e-01],
    [1.6300928e-01, 6.8812118e-01, 4.9853149e-01],
    [1.5480954e-01, 6.8102715e-01, 5.0359681e-01],
    [1.4744706e-01, 6.7389259e-01, 5.0838376e-01],
    [1.4094989e-01, 6.6672018e-01, 5.1289785e-01],
    [1.3533558e-01, 6.5951518e-01, 5.1714376e-01],
    [1.3061078e-01, 6.5228082e-01, 5.2112698e-01],
    [1.2676798e-01, 6.4502107e-01, 5.2485422e-01],
    [1.2378455e-01, 6.3773990e-01, 5.2833094e-01],
    [1.2162122e-01, 6.3043955e-01, 5.3156535e-01],
    [1.2022735e-01, 6.2312449e-01, 5.3456292e-01],
    [1.1954060e-01, 6.1579647e-01, 5.3733333e-01],
    [1.1948792e-01, 6.0845895e-01, 5.3988355e-01],
    [1.1999482e-01, 6.0111447e-01, 5.4221980e-01],
    [1.2098567e-01, 5.9376446e-01, 5.4435422e-01],
    [1.2238718e-01, 5.8641119e-01, 5.4629328e-01],
    [1.2412819e-01, 5.7905692e-01, 5.4804596e-01],
    [1.2614568e-01, 5.7170237e-01, 5.4962190e-01],
    [1.2838753e-01, 5.6434853e-01, 5.5103106e-01],
    [1.3080273e-01, 5.5699668e-01, 5.5228167e-01],
    [1.3335075e-01, 5.4964652e-01, 5.5338307e-01],
    [1.3599751e-01, 5.4229865e-01, 5.5434413e-01],
    [1.3872040e-01, 5.3495208e-01, 5.5517656e-01],
    [1.4149610e-01, 5.2760720e-01, 5.5588684e-01],
    [1.4430921e-01, 5.2026194e-01, 5.5648401e-01],
    [1.4714812e-01, 5.1291637e-01, 5.5697633e-01],
    [1.5000827e-01, 5.0556745e-01, 5.5737141e-01],
    [1.5288767e-01, 4.9821388e-01, 5.5767887e-01],
    [1.5578500e-01, 4.9085367e-01, 5.5790400e-01],
    [1.5869992e-01, 4.8348316e-01, 5.5805276e-01],
    [1.6163808e-01, 4.7610104e-01, 5.5813147e-01],
    [1.6460211e-01, 4.6870331e-01, 5.5814342e-01],
    [1.6759933e-01, 4.6128635e-01, 5.5809491e-01],
    [1.7063600e-01, 4.5384718e-01, 5.5798794e-01],
    [1.7371908e-01, 4.4638116e-01, 5.5782519e-01],
    [1.7685224e-01, 4.3888463e-01, 5.5760829e-01],
    [1.8004182e-01, 4.3135395e-01, 5.5733598e-01],
    [1.8329271e-01, 4.2378415e-01, 5.5700888e-01],
    [1.8660698e-01, 4.1617063e-01, 5.5662375e-01],
    [1.8998799e-01, 4.0851004e-01, 5.5617822e-01],
    [1.9343698e-01, 4.0079740e-01, 5.5566809e-01],
    [1.9695346e-01, 3.9302875e-01, 5.5508698e-01],
    [2.0053668e-01, 3.8519848e-01, 5.5442911e-01],
    [2.0418143e-01, 3.7730286e-01, 5.5368673e-01],
    [2.0788352e-01, 3.6933869e-01, 5.5285007e-01],
    [2.1163820e-01, 3.6130060e-01, 5.5190840e-01],
    [2.1543770e-01, 3.5318462e-01, 5.5085198e-01],
    [2.1927109e-01, 3.4498935e-01, 5.4966790e-01],
    [2.2312782e-01, 3.3670982e-01, 5.4834082e-01],
    [2.2699598e-01, 3.2834445e-01, 5.4685707e-01],
    [2.3086206e-01, 3.1989143e-01, 5.4519990e-01],
    [2.3471155e-01, 3.1134860e-01, 5.4335482e-01],
    [2.3853171e-01, 3.0271313e-01, 5.4130436e-01],
    [2.4807273e-01, 2.8006392e-01, 5.3489622e-01],
    [2.5842674e-01, 2.5292079e-01, 5.2521515e-01],
    [2.6747955e-01, 2.2502763e-01, 5.1266117e-01],
    [2.7478128e-01, 1.9647556e-01, 4.9686516e-01],
    [2.7995511e-01, 1.6735126e-01, 4.7759222e-01],
    [2.8271933e-01, 1.3766737e-01, 4.5477408e-01],
    [2.8290210e-01, 1.0723970e-01, 4.2850816e-01],
    [2.8042859e-01, 7.5399247e-02, 3.9902739e-01],
    [2.7529395e-01, 4.0218357e-02, 3.6665581e-01],
    [2.6548575e-01, 9.3873293e-03, 3.3142864e-01],
    [2.3672010e-01, 4.4848764e-03, 2.9201340e-01],
    [2.0426851e-01, 4.0678996e-03, 2.5193463e-01],
    [1.7181692e-01, 3.6509229e-03, 2.1185586e-01],
    [1.3936532e-01, 3.2339462e-03, 1.7177709e-01],
    [1.0691373e-01, 2.8169695e-03, 1.3169832e-01],
    [7.4462142e-02, 2.3999928e-03, 9.1619547e-02],
    [4.2010550e-02, 1.9830161e-03, 5.1540777e-02],
    [9.5589589e-03, 1.5660394e-03, 1.1462007e-02],
    [2.3768980e-03, 1.3540039e-03, 1.9032161e-02],
    [8.8012235e-03, 7.0267059e-03, 5.0862518e-02],
    [1.9446294e-02, 1.5127667e-02, 8.8637392e-02],
    [3.7697282e-02, 2.5859600e-02, 1.3202545e-01],
    [5.8728318e-02, 3.5382588e-02, 1.7261904e-01],
    [8.1791620e-02, 4.3173322e-02, 2.1489589e-01],
    [1.1029494e-01, 4.7276573e-02, 2.6239396e-01],
    [1.3859700e-01, 4.6434176e-02, 3.0316306e-01],
    [1.6913087e-01, 4.2549733e-02, 3.4019953e-01],
    [2.0368196e-01, 3.7766039e-02, 3.7263815e-01],
    [2.3403307e-01, 3.6703235e-02, 3.9357891e-01],
    [2.6715736e-01, 4.0291439e-02, 4.1008191e-01],
    [2.9653780e-01, 4.7336078e-02, 4.2023573e-01],
    [3.2824607e-01, 5.7612682e-02, 4.2733684e-01],
    [3.5608337e-01, 6.7701565e-02, 4.3100445e-01],
    [3.8447011e-01, 7.8305894e-02, 4.3289029e-01],
    [4.1552632e-01, 8.9900529e-02, 4.3293085e-01],
    [4.4303814e-01, 9.9993275e-02, 4.3137425e-01],
    [4.6876216e-01, 1.0927308e-01, 4.2878689e-01],
    [4.7855800e-01, 1.1276400e-01, 4.2747500e-01],
    [4.8740442e-01, 1.1589924e-01, 4.2613007e-01],
    [4.9984922e-01, 1.2029185e-01, 4.2401744e-01],
    [5.0606118e-01, 1.2247841e-01, 4.2286188e-01],
    [5.1851153e-01, 1.2685318e-01, 4.2034131e-01],
    [5.2472566e-01, 1.2903499e-01, 4.1898068e-01],
    [5.3717780e-01, 1.3341200e-01, 4.1604940e-01],
    [5.4715700e-01, 1.3692900e-01, 4.1351100e-01],
    [5.5583592e-01, 1.4000145e-01, 4.1114233e-01],
    [5.6585400e-01, 1.4356700e-01, 4.0825800e-01],
    [5.7447258e-01, 1.4665863e-01, 4.0561651e-01],
    [5.8688438e-01, 1.5115711e-01, 4.0158808e-01],
    [5.9307038e-01, 1.5342324e-01, 3.9947494e-01],
    [6.0544545e-01, 1.5801643e-01, 3.9503339e-01],
    [6.1551300e-01, 1.6181700e-01, 3.9121900e-01],
    [6.2393232e-01, 1.6505601e-01, 3.8786741e-01],
    [6.3399800e-01, 1.6899200e-01, 3.8370400e-01],
    [6.4232078e-01, 1.7232159e-01, 3.8009933e-01],
    [6.5451982e-01, 1.7731782e-01, 3.7458435e-01],
    [6.6058399e-01, 1.7986142e-01, 3.7173515e-01],
    [6.7268305e-01, 1.8506889e-01, 3.6583335e-01],
    [6.7869120e-01, 1.8772613e-01, 3.6279408e-01],
    [6.9066776e-01, 1.9317724e-01, 3.5651873e-01],
    [7.0057600e-01, 1.9785100e-01, 3.5111300e-01],
    [7.0844221e-01, 2.0169252e-01, 3.4665898e-01],
    [7.2016454e-01, 2.0762354e-01, 3.3978951e-01],
    [7.2597003e-01, 2.1066333e-01, 3.3627825e-01],
    [7.3750629e-01, 2.1692438e-01, 3.2907955e-01],
    [7.4321249e-01, 2.2013663e-01, 3.2540894e-01],
    [7.5453758e-01, 2.2675655e-01, 3.1790186e-01],
    [7.6401000e-01, 2.3255400e-01, 3.1139900e-01],
    [7.7121769e-01, 2.3716104e-01, 3.0628954e-01],
    [7.8212999e-01, 2.4443839e-01, 2.9831713e-01],
    [7.8750665e-01, 2.4817365e-01, 2.9427494e-01],
    [7.9813459e-01, 2.5587311e-01, 2.8605533e-01],
    [8.0336278e-01, 2.5982425e-01, 2.8189636e-01],
    [8.1367956e-01, 2.6796285e-01, 2.7345507e-01],
    [8.2238600e-01, 2.7519700e-01, 2.6608500e-01],
    [8.2872451e-01, 2.8072375e-01, 2.6055122e-01],
    [8.3716500e-01, 2.8838500e-01, 2.5298800e-01],
    [8.4323300e-01, 2.9416660e-01, 2.4737707e-01],
    [8.5259473e-01, 3.0351540e-01, 2.3845286e-01],
    [8.5716732e-01, 3.0829433e-01, 2.3396131e-01],
    [8.6612833e-01, 3.1809778e-01, 2.2488933e-01],
    [8.7374100e-01, 3.2690600e-01, 2.1688600e-01],
    [8.7903544e-01, 3.3335138e-01, 2.1112165e-01],
    [8.8630200e-01, 3.4258600e-01, 2.0296800e-01],
    [8.9128925e-01, 3.4925881e-01, 1.9716121e-01],
    [8.9908871e-01, 3.6022418e-01, 1.8774376e-01],
    [9.0286648e-01, 3.6579879e-01, 1.8301328e-01],
    [9.1020336e-01, 3.7717067e-01, 1.7346942e-01],
    [9.1374746e-01, 3.8294253e-01, 1.6867513e-01],
    [9.2061161e-01, 3.9470006e-01, 1.5899897e-01],
    [9.2647000e-01, 4.0538900e-01, 1.5029200e-01],
    [9.3029899e-01, 4.1278093e-01, 1.4431876e-01],
    [9.3635556e-01, 4.2513210e-01, 1.3440037e-01],
    [9.3925536e-01, 4.3137856e-01, 1.2941248e-01],
    [9.4482100e-01, 4.4405580e-01, 1.1933760e-01],
    [9.4747422e-01, 4.5045943e-01, 1.1427043e-01],
    [9.5254213e-01, 4.6343970e-01, 1.0404005e-01],
    [9.5685200e-01, 4.7535600e-01, 9.4695000e-02],
    [9.5951562e-01, 4.8324927e-01, 8.8538725e-02],
    [9.6338700e-01, 4.9546200e-01, 7.9073000e-02],
    [9.6573671e-01, 5.0345412e-01, 7.2941353e-02],
    [9.6946600e-01, 5.1713780e-01, 6.2604376e-02],
    [9.7120078e-01, 5.2402412e-01, 5.7516612e-02],
    [9.7442322e-01, 5.3793591e-01, 4.7605694e-02],
    [9.7709200e-01, 5.5085000e-01, 3.9050000e-02],
    [9.7861714e-01, 5.5905386e-01, 3.4378529e-02],
    [9.8082400e-01, 5.7220900e-01, 2.8508000e-02],
    [9.8204193e-01, 5.8046703e-01, 2.6013208e-02],
    [9.8389682e-01, 5.9490430e-01, 2.3746204e-02],
    [9.8469321e-01, 6.0215259e-01, 2.3690141e-02],
    [9.8602749e-01, 6.1675951e-01, 2.5896980e-02],
    [9.8656360e-01, 6.2408900e-01, 2.8226533e-02],
    [9.8737388e-01, 6.3885153e-01, 3.5559176e-02],
    [9.8781900e-01, 6.5277300e-01, 4.5581000e-02],
    [9.8792831e-01, 6.6116152e-01, 5.2549800e-02],
    [9.8785518e-01, 6.7615406e-01, 6.6107824e-02],
    [9.8768557e-01, 6.8366665e-01, 7.3342055e-02],
    [9.8707678e-01, 6.9877807e-01, 8.8605369e-02],
    [9.8663905e-01, 7.0634608e-01, 9.6558953e-02],
    [9.8549453e-01, 7.2156149e-01, 1.1310138e-01],
    [9.8407500e-01, 7.3608700e-01, 1.2952700e-01],
    [9.8310489e-01, 7.4448120e-01, 1.3931060e-01],
    [9.8117300e-01, 7.5913500e-01, 1.5686300e-01],
    [9.7992623e-01, 7.6750200e-01, 1.6718848e-01],
    [9.7738261e-01, 7.8289344e-01, 1.8675435e-01],
    [9.7599271e-01, 7.9057933e-01, 1.9682694e-01],
    [9.7296729e-01, 8.0598369e-01, 2.1768029e-01],
    [9.6978300e-01, 8.2082500e-01, 2.3868600e-01],
    [9.6792113e-01, 8.2902673e-01, 2.5074280e-01],
    [9.6439400e-01, 8.4384800e-01, 2.7339100e-01],
    [9.6240576e-01, 8.5192265e-01, 2.8627918e-01],
    [9.5861645e-01, 8.6703604e-01, 3.1154218e-01],
    [9.5674035e-01, 8.7450829e-01, 3.2466229e-01],
    [9.5313646e-01, 8.8928478e-01, 3.5203996e-01],
    [9.5148009e-01, 8.9653585e-01, 3.6625870e-01],
    [9.4864029e-01, 9.1074461e-01, 3.9589198e-01],
    [9.4680900e-01, 9.2416800e-01, 4.2637300e-01],
    [9.4639235e-01, 9.3096172e-01, 4.4287602e-01],
    [9.4693138e-01, 9.4351188e-01, 4.7542173e-01],
    [9.4797484e-01, 9.4945318e-01, 4.9181268e-01],
    [9.5179469e-01, 9.6069110e-01, 5.2451982e-01],
    [9.5458182e-01, 9.6597611e-01, 5.4061063e-01],
    [9.6186420e-01, 9.7597993e-01, 5.7210478e-01],
    [9.7116200e-01, 9.8528200e-01, 6.0215400e-01],
    [9.7653353e-01, 9.8977008e-01, 6.1681591e-01],
    [9.8836200e-01, 9.9836400e-01, 6.4492400e-01],
]


def virino():
    """
    Defines the virino colormap, useful for plotting angular data.

    Returns:
        The virino colormap
    """

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("virino", _virino_list)
    return cmap


def plot_DoLP_AoLP(DoLP, AoLP, filename=None):
    plt.figure(figsize=(12, 6), layout="constrained")
    for i, colour in enumerate(["R", "G1", "G2", "B"]):
        plt.subplot(2, 4, i + 1)
        plt.imshow(DoLP[:, :, i])  # , vmin=0, vmax=0.05)
        plt.colorbar()
        plt.title(colour + " DoLP")
        plt.subplot(2, 4, i + 5)
        plt.imshow(AoLP[:, :, i], cmap=virino)
        plt.colorbar()
        plt.title(colour + " AoLP")

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_fringe_pattern(intensity, isoclinic, filename="output.png"):
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(intensity)
    plt.colorbar()
    plt.title("Fringe Pattern")

    plt.subplot(1, 2, 2)
    plt.imshow(isoclinic, cmap=virino)
    plt.colorbar()
    plt.title("Isoclinic Angle")

    plt.savefig(filename)


def show_all_channels(data, metadata, filename=None):
    if metadata["dtype"] == "uint16":
        data = data.astype("float32") / 65535.0

    plt.figure(figsize=(12, 12), layout="constrained")
    # work with both RGB and RGGB
    ncolour_channels = data.shape[2]
    if ncolour_channels == 3:
        colours = ["R", "G", "B"]
    else:
        colours = ["R", "G1", "G2", "B"]

    for i, colour in enumerate(colours):
        for j, polarisation in enumerate(["0", "90", "45", "135"]):
            plt.subplot(4, 4, i * 4 + j + 1)
            plt.title(f"{colour}_{polarisation}")
            plt.imshow(data[:, :, i, j])
            plt.axis("off")

    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def plot_optimization_history(history, S_m_hat, filename=None):
    """
    Plot the evolution of stress components and Stokes parameters during optimization.

    Parameters
    ----------
    history : dict
        Dictionary containing optimization history with keys:
        - 'all_paths': list of dicts, each containing optimization path data
        - 'best_path_index': index of the path that led to the best solution
    S_m_hat : ndarray
        Measured normalised Stokes components, shape (3, 2) for RGB channels.
        Used to show target values.
    filename : str, optional
        If provided, save figure to this filename instead of displaying.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    all_paths = history["all_paths"]
    best_path_index = history["best_path_index"]

    colors_rgb = ["red", "green", "blue"]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Stress components evolution (all paths)
    ax1 = fig.add_subplot(gs[0, :])
    for i, path in enumerate(all_paths):
        stress_params = path["stress_params"]
        n_iter = len(stress_params)
        iterations = np.arange(n_iter)
        is_best = path["is_best"]
        alpha = 0.8 if is_best else 0.1
        linewidth = 2 if is_best else 0.1
        linestyle = "-" if is_best else "-"
        markersize = 2 if is_best else 0.1

        ax1.plot(
            iterations,
            stress_params[:, 0],
            "o-",
            alpha=alpha,
            markersize=markersize,
            linewidth=linewidth,
            color="C0",
        )
        ax1.plot(
            iterations,
            stress_params[:, 1],
            "s-",
            alpha=alpha,
            markersize=markersize,
            linewidth=linewidth,
            color="C1",
        )
        ax1.plot(
            iterations,
            stress_params[:, 2],
            "^-",
            alpha=alpha,
            markersize=markersize,
            linewidth=linewidth,
            color="C2",
        )

    # Add legend (using best path)
    best_path = all_paths[best_path_index]
    ax1.plot([], [], "o-", color="C0", label="σ_xx", linewidth=2)
    ax1.plot([], [], "s-", color="C1", label="σ_yy", linewidth=2)
    ax1.plot([], [], "^-", color="C2", label="σ_xy", linewidth=2)
    ax1.set_xlabel("Iteration (within each path)")
    ax1.set_ylabel("Stress (Pa)")
    ax1.set_title(
        f"Evolution of Stress Components ({len(all_paths)} optimization paths, best path highlighted)"
    )
    ax1.legend()
    # ax1.grid(True, alpha=0.3)

    # Plot 2: Residual evolution (all paths)
    ax2 = fig.add_subplot(gs[1, 0])
    for i, path in enumerate(all_paths):
        residuals = path["residuals"]
        n_iter = len(residuals)
        iterations = np.arange(n_iter)
        is_best = path["is_best"]
        alpha = 0.8 if is_best else 0.2
        linewidth = 2 if is_best else 0.5

        ax2.semilogy(
            iterations, residuals, "o-", alpha=alpha, markersize=2 if is_best else 1, linewidth=linewidth
        )
    ax2.set_xlabel("Iteration (within each path)")
    ax2.set_ylabel("Residual (log scale)")
    ax2.set_title("Residual Evolution (All Paths)")
    ax2.grid(True, alpha=0.3)

    # Plot 3: S1_hat evolution for RGB channels (best path only)
    ax3 = fig.add_subplot(gs[1, 1])
    S_predicted = best_path["S_predicted"]
    n_iter = len(S_predicted)
    iterations = np.arange(n_iter)
    for c, color in enumerate(colors_rgb):
        ax3.plot(
            iterations,
            S_predicted[:, c, 0],
            "o-",
            color=color,
            alpha=0.7,
            markersize=2,
            label=f"{color.upper()} predicted",
        )
        ax3.axhline(
            S_m_hat[c, 0],
            color=color,
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"{color.upper()} measured",
        )
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("S1_hat (normalised)")
    ax3.set_title("S1_hat Evolution - Best Path (RGB Channels)")
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)

    # Plot 4: S2_hat evolution for RGB channels (best path only)
    ax4 = fig.add_subplot(gs[1, 2])
    for c, color in enumerate(colors_rgb):
        ax4.plot(
            iterations,
            S_predicted[:, c, 1],
            "s-",
            color=color,
            alpha=0.7,
            markersize=2,
            label=f"{color.upper()} predicted",
        )
        ax4.axhline(
            S_m_hat[c, 1],
            color=color,
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"{color.upper()} measured",
        )
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("S2_hat (normalised)")
    ax4.set_title("S2_hat Evolution - Best Path (RGB Channels)")
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)

    # Plot 5: 3D trajectory in stress space (all paths)
    ax5 = fig.add_subplot(gs[2, 0], projection="3d")
    for i, path in enumerate(all_paths):
        stress_params = path["stress_params"]
        n_iter = len(stress_params)
        iterations_path = np.arange(n_iter)
        is_best = path["is_best"]
        alpha = 0.7 if is_best else 0.15
        linewidth = 2 if is_best else 0.5
        s = 20 if is_best else 5

        scatter = ax5.scatter(
            stress_params[:, 0],
            stress_params[:, 1],
            stress_params[:, 2],
            c=iterations_path,
            cmap="viridis",
            s=s,
            alpha=alpha,
        )
        ax5.plot(
            stress_params[:, 0],
            stress_params[:, 1],
            stress_params[:, 2],
            "k-",
            alpha=alpha,
            linewidth=linewidth,
        )

        # Mark start and end points for best path only
        if is_best:
            ax5.scatter(
                stress_params[0, 0],
                stress_params[0, 1],
                stress_params[0, 2],
                color="green",
                s=100,
                marker="o",
                label="Start (best)",
                edgecolors="black",
            )
            ax5.scatter(
                stress_params[-1, 0],
                stress_params[-1, 1],
                stress_params[-1, 2],
                color="red",
                s=100,
                marker="*",
                label="End (best)",
                edgecolors="black",
            )

    ax5.set_xlabel("σ_xx (Pa)")
    ax5.set_ylabel("σ_yy (Pa)")
    ax5.set_zlabel("σ_xy (Pa)")
    ax5.set_title(f"Optimization Trajectories in Stress Space ({len(all_paths)} paths)")
    # ax5.legend()
    plt.colorbar(scatter, ax=ax5, label="Iteration", shrink=0.6)

    # Plot 6: Final comparison of measured vs predicted (best path)
    ax6 = fig.add_subplot(gs[2, 1:])
    x_pos = np.arange(6)
    measured = np.concatenate([S_m_hat[:, 0], S_m_hat[:, 1]])
    S_predicted_best = best_path["S_predicted"]
    predicted = np.concatenate([S_predicted_best[-1, :, 0], S_predicted_best[-1, :, 1]])

    width = 0.35
    ax6.bar(x_pos - width / 2, measured, width, label="Measured", alpha=0.7, color="steelblue")
    ax6.bar(x_pos + width / 2, predicted, width, label="Predicted (final)", alpha=0.7, color="coral")
    ax6.set_xlabel("Stokes Component")
    ax6.set_ylabel("Normalised Value")
    ax6.set_title("Final Comparison: Measured vs Predicted Stokes Components (Best Path)")
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(["R S1", "G S1", "B S1", "R S2", "G S2", "B S2"])
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis="y")

    # Add summary text
    final_residual = best_path["residuals"][-1]
    final_stress = best_path["stress_params"][-1]
    fig.text(
        0.02,
        0.98,
        f"Final residual: {final_residual:.2e}\n"
        f"Final σ_xx: {final_stress[0]:.2e} Pa\n"
        f"Final σ_yy: {final_stress[1]:.2e} Pa\n"
        f"Final σ_xy: {final_stress[2]:.2e} Pa\n"
        f"Iterations: {n_iter}",
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()

    return fig


def plot_optimization_history_live(history, S_m_hat, fig=None, axes=None):
    """
    Create or update a live plot of optimization history (for interactive use).

    Parameters
    ----------
    history : dict
        Dictionary containing optimization history (same format as plot_optimization_history).
    S_m_hat : ndarray
        Measured normalised Stokes components, shape (3, 2).
    fig : matplotlib.figure.Figure, optional
        Existing figure to update. If None, creates new figure.
    axes : list, optional
        List of existing axes to update. If None, creates new axes.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : list
        List of axes objects.
    """
    stress_params = history["stress_params"]
    S_predicted = history["S_predicted"]
    residuals = history["residuals"]
    n_iter = len(residuals)
    iterations = np.arange(n_iter)

    colors_rgb = ["red", "green", "blue"]

    # Create figure if needed
    if fig is None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle("Live Optimization Progress", fontsize=14, fontweight="bold")
        plt.ion()  # Enable interactive mode
    else:
        # Clear existing axes
        for ax in axes.flat:
            ax.clear()

    axes = axes.flatten()

    # Plot 1: Stress components
    axes[0].plot(iterations, stress_params[:, 0], "o-", label="σ_xx", alpha=0.7, markersize=3)
    axes[0].plot(iterations, stress_params[:, 1], "s-", label="σ_yy", alpha=0.7, markersize=3)
    axes[0].plot(iterations, stress_params[:, 2], "^-", label="σ_xy", alpha=0.7, markersize=3)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Stress (Pa)")
    axes[0].set_title("Stress Components")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Residual
    axes[1].semilogy(iterations, residuals, "ko-", markersize=3)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Residual (log)")
    axes[1].set_title("Residual Evolution")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: S1_hat for RGB
    for c, color in enumerate(colors_rgb):
        axes[2].plot(
            iterations,
            S_predicted[:, c, 0],
            "o-",
            color=color,
            alpha=0.7,
            markersize=2,
            label=f"{color[0].upper()} pred",
        )
        axes[2].axhline(S_m_hat[c, 0], color=color, linestyle="--", linewidth=2, alpha=0.5)
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("S1_hat")
    axes[2].set_title("S1_hat (RGB)")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    # Plot 4: S2_hat for RGB
    for c, color in enumerate(colors_rgb):
        axes[3].plot(
            iterations,
            S_predicted[:, c, 1],
            "s-",
            color=color,
            alpha=0.7,
            markersize=2,
            label=f"{color[0].upper()} pred",
        )
        axes[3].axhline(S_m_hat[c, 1], color=color, linestyle="--", linewidth=2, alpha=0.5)
    axes[3].set_xlabel("Iteration")
    axes[3].set_ylabel("S2_hat")
    axes[3].set_title("S2_hat (RGB)")
    axes[3].legend(fontsize=7)
    axes[3].grid(True, alpha=0.3)

    # Plot 5: Trajectory projection (sigma_xx vs sigma_yy)
    axes[4].plot(stress_params[:, 0], stress_params[:, 1], "o-", markersize=3, alpha=0.7, color="navy")
    axes[4].scatter(
        stress_params[0, 0],
        stress_params[0, 1],
        color="green",
        s=100,
        marker="o",
        label="Start",
        edgecolors="black",
        zorder=5,
    )
    axes[4].scatter(
        stress_params[-1, 0],
        stress_params[-1, 1],
        color="red",
        s=100,
        marker="*",
        label="End",
        edgecolors="black",
        zorder=5,
    )
    axes[4].set_xlabel("σ_xx (Pa)")
    axes[4].set_ylabel("σ_yy (Pa)")
    axes[4].set_title("Stress Trajectory (xy plane)")
    axes[4].legend(fontsize=8)
    axes[4].grid(True, alpha=0.3)

    # Plot 6: Final comparison
    x_pos = np.arange(6)
    measured = np.concatenate([S_m_hat[:, 0], S_m_hat[:, 1]])
    predicted = np.concatenate([S_predicted[-1, :, 0], S_predicted[-1, :, 1]])

    width = 0.35
    axes[5].bar(x_pos - width / 2, measured, width, label="Measured", alpha=0.7, color="steelblue")
    axes[5].bar(x_pos + width / 2, predicted, width, label="Predicted", alpha=0.7, color="coral")
    axes[5].set_xlabel("Component")
    axes[5].set_ylabel("Value")
    axes[5].set_title("Measured vs Predicted")
    axes[5].set_xticks(x_pos)
    axes[5].set_xticklabels(["RS1", "GS1", "BS1", "RS2", "GS2", "BS2"], fontsize=8)
    axes[5].legend(fontsize=8)
    axes[5].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig, axes.reshape(2, 3)
