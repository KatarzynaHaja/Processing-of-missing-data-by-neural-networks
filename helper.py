# from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
# X = input_data.read_data_sets("./data_mnist/", one_hot=True)
# image = X.train.images[0]
# plt.imshow(image.reshape(28, 28), origin="upper", cmap="gray")
# plt.axis('off')
# plt.savefig('mnist_example',  bbox_inches='tight', pad_inches=0)
#
# new_image = 1- image
# plt.imshow(new_image.reshape(28,28), origin="upper", cmap="gray")
# plt.axis('off')
# plt.savefig('mnist_example_changed_background',  bbox_inches='tight', pad_inches=0)
#
# import numpy as np
#
#
# def random_mask_mnist(width_window, margin=0):
#     margin_left = margin
#     margin_righ = margin
#     margin_top = margin
#     margin_bottom = margin
#     start_width = margin_top + np.random.randint(28 - width_window - margin_top - margin_bottom)
#     start_height = margin_left + np.random.randint(28 - width_window - margin_left - margin_righ)
#
#     return np.concatenate([28 * i + np.arange(start_height, start_height + width_window) for i in
#                            np.arange(start_width, start_width + width_window)], axis=0).astype(np.int32)
#
#
# def data_with_mask_mnist(x, width_window=10):
#     h = width_window
#     if width_window <= 0:
#         h = np.random.randint(8, 20)
#     mask = random_mask_mnist(h)
#     x[mask] = 1
#     return x
#
# image_with_patch = data_with_mask_mnist(new_image, 13)
#
# plt.imshow(new_image.reshape(28,28), origin="upper", cmap="gray")
# plt.axis('off')
# plt.savefig('mnist_example_with_patch',  bbox_inches='tight', pad_inches=0)

#
# from processing_images import DatasetProcessor
# import matplotlib.pyplot as plt
#
# d = DatasetProcessor('svhn')
# data_train, data_test = d.load_data()
# data_train['X'] = d.reshape_data(data_train['X'])
# data_test['X'] = d.reshape_data(data_test['X'])
# data_train, data_test = d.mask_data(data_train, data_test)
#
# plt.imshow(data_train['X'][0].reshape(32, 32,3).astype('uint8'), origin="upper")
# plt.axis('off')
# plt.savefig('svhn_masked_example',  bbox_inches='tight', pad_inches=0)

import numpy as np

# data = [0.0052325632, 0.008187881, 0.014886782]
# labels = ['metoda analiczna', 'ostatnia warstwa', 'imputacja']
# pos = np.arange(len(data))
#
# plt.bar(pos,data,color='lightsteelblue', width=0.3)
# plt.xticks(pos, labels)
# plt.xlabel('Metody', fontsize=12)
# plt.ylabel('Średni koszt', fontsize=12)
# plt.title('Porównianie średniego kosztu - autoenkoder liniowy', fontsize=13)
# plt.savefig("Porównianie średniego kosztu")
# plt.show()


data = [0.9255, 0.9281,0.935, 0.89641]
labels = ['pierwsza warstwa', 'ostatnia warstwa', 'imputacja', 'uśredniony koszt']
pos = np.arange(len(data))

plt.bar(pos,data,color='lightsteelblue', width=0.3)
plt.xticks(pos, labels)
plt.xlabel('Metody', fontsize=12)
plt.ylabel('Precyzja', fontsize=12)
plt.title('Porównianie precyzji - klasyfikator konwolucyjny', fontsize=12)
plt.savefig("Porównianie precyzji klasyfikator konwolucyjny")
plt.show()

# results = {
#     'imputacja': [0.7722316524827022, 0.37674226482183054, 0.3058652415463963, 0.2694526812944201, 0.2447639182654488, 0.22603588800105387, 0.21046609932294374, 0.19702684020838235, 0.18518628844041574, 0.17453090180843478, 0.1645293257419319, 0.155730656499956, 0.14743203179622383, 0.13997599449143144, 0.13291155158192985, 0.12628417928259694, 0.1201992287008879, 0.1140435828902673, 0.10816057020748453, 0.10265899009797158, 0.09723002802660825, 0.09217453497051065, 0.08737702832022753, 0.08266539533315737, 0.07820721952557971, 0.0739597635045111, 0.06988245677613579, 0.06595601557041259, 0.06220301852594872, 0.05875869198876153, 0.055423984524161835, 0.05198283444547146, 0.04883435880379075, 0.04601700449479558, 0.043098082585350805, 0.040400264063459125, 0.03771421956310229, 0.035220886567678064, 0.03291214903244939, 0.030576174686910963, 0.028513774110037814, 0.02647709263554312, 0.02453768328727444, 0.0228943128326956, 0.021255048773166956, 0.019609444746798942, 0.01809694502260837, 0.01679214790198718, 0.015548655346078275, 0.014469024563073953],
#     'pierwsza warstwa': [1.0362647606257231, 0.5527922304583511, 0.43270646083809444, 0.37280858082998203, 0.333112555340158, 0.3101094710283909, 0.29151845050322994, 0.27654412916587173, 0.26095677804249545, 0.24981828801316558, 0.24106344935723836, 0.22810355675047886, 0.22129522875109656, 0.21308766804934762, 0.20620466466251505, 0.20003788327479494, 0.19452949692424695, 0.18930255208160116, 0.18321927849472228, 0.17909131992209326, 0.17204549746849163, 0.16909328083793942, 0.16332565706951827, 0.16006513257354837, 0.156625938809415, 0.1499619689182952, 0.14664900690404975, 0.1439350243550492, 0.14058431176028546, 0.13698513154238592, 0.13480714049647813, 0.12987177056920138, 0.12748284810884045, 0.12273995631197696, 0.12197610284543321, 0.11672609181379698, 0.11601870870193287, 0.11205725445915166, 0.11014145182061298, 0.10780695541051669, 0.10494449916398706, 0.10016390373294726, 0.09884923976172212, 0.099094241177197, 0.0956235608810096, 0.09282935140604301, 0.09180750405701578, 0.08987298939725095, 0.08770027907421038, 0.08561007187077536],
#     'ostatnia warstwa': [1.0335481299978613, 0.6087661697523835, 0.4749616991134406, 0.39728732699608565, 0.35291882358031723, 0.32275417861252226, 0.30128556121535366, 0.28383048842048514, 0.2689787204918366, 0.2568453183967937, 0.24495808256419246, 0.23586923213907635, 0.22582934238341865, 0.21811253520008503, 0.21126121343075535, 0.20244330040160763, 0.19615125017537713, 0.19060049730985143, 0.1825978475361542, 0.17822664851817163, 0.17353869163527721, 0.16865579570330272, 0.1633890530824167, 0.15764310058993444, 0.15356504669026996, 0.15038684086372145, 0.14639350643047544, 0.14223614903069312, 0.1376293496130093, 0.13461231524195214, 0.13132466694635012, 0.1275561580069068, 0.12331338161743562, 0.121330509087498, 0.11810314613237231, 0.11451489597611017, 0.11175016980163961, 0.10943094094400972, 0.10527116099315044, 0.1046575165418225, 0.09993843740688223, 0.09771538000473616, 0.09434137435286274, 0.09269630921569819, 0.08979106343811274, 0.0884100375782397, 0.08564796794882983, 0.0840905519092306, 0.08237864999374325, 0.07787975054602893],
#     'uśredniony koszt': [1.4519327692470672, 0.9386279607089768, 0.7516304096326284, 0.6416880050641555, 0.5727311388282615, 0.5236793843957751, 0.4839260052971984, 0.45384491295298957, 0.4275449440273542, 0.4060583563870615, 0.38607369059733654, 0.3710841266644362, 0.3562157216012409, 0.34343436225051793, 0.3315874517215492, 0.3217335076388991, 0.31297837727734595, 0.30489325983508364, 0.29677858667467677, 0.2895836045985068, 0.2827934868789871, 0.2755543921658893, 0.2704764470453694, 0.2636738418275111, 0.2579681200514156, 0.25217612756999497, 0.2470382082973183, 0.24279222551763613, 0.23740620863633674, 0.2326032620777698, 0.22903776638496046, 0.22502791291759788, 0.220506522314443, 0.21598248982344503, 0.2127067501796572, 0.20884507423108475, 0.20474488717071251, 0.20200289991425135, 0.1973471634113799, 0.1952711583101066, 0.19053539073114242, 0.1875184455168747, 0.18354154494401323, 0.1825210229830862, 0.18032102617711626, 0.17564046931289337, 0.17328361872414494, 0.17178822733891197, 0.16790603576672541, 0.16529497718037353]
# }
#
# plt.figure(figsize=(9,8))
# for key, data in results.items():
#     plt.plot([x for x in range(1, 51)], data, label=key)
#     plt.xlabel('epoki')
#     plt.ylabel('koszt')
# plt.legend()
# plt.savefig('Training_loss_classifier_cnn.png')
