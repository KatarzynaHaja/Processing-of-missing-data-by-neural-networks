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


# data = [0.9255, 0.9281,0.935, 0.89641]
# labels = ['pierwsza warstwa', 'ostatnia warstwa', 'imputacja', 'uśredniony koszt']
# pos = np.arange(len(data))
#
# plt.bar(pos,data,color='lightsteelblue', width=0.3)
# plt.xticks(pos, labels)
# plt.xlabel('Metody', fontsize=12)
# plt.ylabel('Precyzja', fontsize=12)
# plt.title('Porównianie precyzji - klasyfikator konwolucyjny', fontsize=12)
# plt.savefig("Porównianie precyzji klasyfikator konwolucyjny")
# plt.show()

# results = {
#     'imputacja': [0.19657721525995164, 0.037353277568978244, 0.02866424496968732, 0.02401254862597863, 0.02128525575715588, 0.019362408197399556, 0.01805502301635335, 0.01713923631132447, 0.016427250510715176, 0.015847552550649326, 0.015380146688325637, 0.014988755542814194, 0.01466138352435933, 0.014379915023688091, 0.014135831773252916, 0.013919614056807585, 0.01372589688972484, 0.013547991124703366, 0.013383006238873646, 0.013236083990466397, 0.013097721963426099, 0.01296744392133419, 0.012839559767100443, 0.012711551502956131, 0.012587318912794467, 0.012464503018721474, 0.0123446631094385, 0.012233935509924066, 0.012118696243520355, 0.011997567821951983, 0.011885428488209024, 0.011775765025401301, 0.011666791934513502, 0.011559943130534118, 0.011461035946775895, 0.011371332943138166, 0.011284728853992851, 0.011197350632289849, 0.011108786591721858, 0.011019890536146349, 0.01093580254093723, 0.010858421859239029, 0.0107852876006718, 0.010717632802046677, 0.010653557535640273, 0.010593261307846253, 0.01053780891504686, 0.010485701889161684, 0.010436704714373643, 0.01039020493079236],
#     'pierwsza warstwa': [0.19400980233266008, 0.03520999265553423, 0.027102507983873862, 0.020855555532498522, 0.01758125325643742, 0.01601341765526772, 0.015010190870507723, 0.014249092346213917, 0.013667859279859246, 0.013228209684382077, 0.012859923519700297, 0.01253070031634971, 0.0122370702357611, 0.011980880024076887, 0.011750097623908367, 0.011538993565196045, 0.011333616483758226, 0.011136016078458074, 0.01096000109214183, 0.01079173702494628, 0.010613728219663287, 0.010446289029099201, 0.01030241771391681, 0.010160737881501436, 0.010036031820708093, 0.009924861772654099, 0.009810383570214404, 0.00970067495763614, 0.009595317471255423, 0.009496390646399976, 0.009398895709002854, 0.009301270670421446, 0.009209528311101814, 0.00912992912839154, 0.009051811691684659, 0.0089777643767402, 0.008906683233014617, 0.008835916774841131, 0.008766181399281077, 0.008701719029743961, 0.008632294154339295, 0.008569184674726674, 0.008500076182905216, 0.008431700614311076, 0.00836744023717838, 0.008306399707194415, 0.008245385553934935, 0.008182663925735625, 0.00812741049901648, 0.008069988001562852],
#     'ostatnia warstwa': [0.27871570618027064, 0.11290812439477337, 0.04695586121299949, 0.030933952428875255, 0.025182298629834343, 0.02265295448637404, 0.021210369518514998, 0.020093817437262185, 0.01918685163003017, 0.01840169614835043, 0.017715497018001455, 0.017126071521541407, 0.01659335090014905, 0.016133255233399077, 0.015723063552777607, 0.015352648753886363, 0.015023137181552909, 0.014720724121707277, 0.014444274394015588, 0.01417222675391285, 0.013915434047516901, 0.013679524511044669, 0.013455266132679814, 0.01324290376625888, 0.013037049740223404, 0.012836061481463379, 0.012650229340564955, 0.012460894126131813, 0.012276125354554145, 0.012098084034949351, 0.011918867738922347, 0.011737956440551513, 0.011547159706371093, 0.011351186990497295, 0.011148513298008048, 0.010948637649709572, 0.010744992963745344, 0.010551517667936565, 0.010367688154143271, 0.01018752254559779, 0.010012253550967055, 0.009841165261341496, 0.009680210464900365, 0.009529747309503008, 0.00938949471619766, 0.009256744475757223, 0.009129281642697459, 0.009008411325244431, 0.008884174739906728, 0.008778383555814758],
#     'uśredniony koszt': [0.2897904329205673, 0.0378249666453327, 0.025830280605854253, 0.021812869949878164, 0.019624430128579954, 0.017843277474601394, 0.015560342878773094, 0.014229886625086738, 0.013590170421705608, 0.013122639406915751, 0.012763983786002056, 0.01246932490653195, 0.01222178789158074, 0.012004655777729543, 0.011816912044136676, 0.011656544663354534, 0.011510250597172285, 0.011377525749565654, 0.011262832676890478, 0.011157552850944014, 0.011062382571513018, 0.01097142473873914, 0.010898366850248847, 0.010824321598196086, 0.01075736734450323, 0.010691389398812458, 0.01062129505110348, 0.010554360109686219, 0.010488237238605338, 0.010428628769093798, 0.0103780212657673, 0.010324236121655143, 0.010274259622295756, 0.010228662191656333, 0.010180429950007475, 0.010125622273321706, 0.010058229800518565, 0.00996790757060814, 0.009860570668515195, 0.009744085414036523, 0.009587706726072598, 0.009304088509078242, 0.009055232090221386, 0.00888601907143889, 0.008789786547075568, 0.00869917383150311, 0.008620092451679722, 0.008557161437263532, 0.008506157511483758, 0.008459371422895181]
# }
#
# plt.figure(figsize=(9,8))
# for key, data in results.items():
#     plt.plot([x for x in range(1, 51)], data, label=key)
#     plt.xlabel('epoki')
#     plt.ylabel('koszt')
#     plt.title('Training loss dla imputacji i metod losujących z 10 próbkami')
# plt.legend()
# plt.savefig('Training_loss_autoencoder_10_cnn.png')


# results = {
#     'imputacja': [0.17022855852882043, 0.04972741872739889, 0.04392279953995797, 0.027291383826224515, 0.022156540445868518, 0.020826638255410323, 0.019781178151248396, 0.019033109779521967, 0.018425980912061376, 0.01792138965341382, 0.017494410501317746, 0.017124366588296587, 0.016799535083424345, 0.0165092362075229, 0.016242987730729744, 0.015976445339695985, 0.015741259194281876, 0.015534879226245123, 0.015358730184891421, 0.015197736080784697, 0.015043912010507456, 0.014898116002875444, 0.014754348103924914, 0.014611716330450289, 0.014471347051666405, 0.014338703495215929, 0.014195117402517209, 0.014058476141070015, 0.013927081406359275, 0.013796420091292193, 0.013685514457558508, 0.013570665656070065, 0.0134452709945497, 0.013334628268274277, 0.01323732972640413, 0.013131121319553638, 0.013030454927742706, 0.012941889061923373, 0.012854064738332947, 0.01276435092273652, 0.0126794180701383, 0.012602585773770896, 0.01252649720960885, 0.01245311932374905, 0.012381067869662891, 0.012300690580354524, 0.012221499067639885, 0.012148311907945242, 0.012082590284202641, 0.012021313065916066, 0.011962711139467477, 0.011903608386649498, 0.011840773294393947, 0.011765328917819286, 0.011690070922682937, 0.011613619561635818, 0.011549442658590514, 0.011490902208644817, 0.011430388567700943, 0.01136597263138196, 0.011306261871652805, 0.01125545848438997, 0.011209018186768768, 0.011163866173818487, 0.01111206119668182, 0.011054954083068993, 0.011004612168309636, 0.010955061205327597, 0.010901021241929284, 0.010844344488359447, 0.01079625953603964, 0.010750448052544588, 0.010701030507008498, 0.010644801036347162, 0.010591015430096324, 0.010538568045913627, 0.010492094052272123, 0.0104465573810145, 0.010401087034719132, 0.010353334818456994, 0.010307873213099538, 0.010265523885372146, 0.010223573772575417, 0.010180630818700891, 0.010140375266567518, 0.010101404528932661, 0.010064641363698754, 0.010028861534876865, 0.009985463122669515, 0.009933992250664509, 0.00987526097890138, 0.009824312463156707, 0.009778477061038052, 0.009739548005033604, 0.009703450837731577, 0.009670843560614964, 0.009641247134490964, 0.009613768428326702, 0.009588236140685005, 0.009563001968125887],
#     'pierwsza warstwa': [0.07720478891823436, 0.027811300515660303, 0.023518664517421205, 0.021632018133267383, 0.02043946835751368, 0.019467339970188794, 0.018556308790614325, 0.01779091934699772, 0.017261163180638164, 0.016825206363440894, 0.016462038869430047, 0.016127457333349856, 0.015824754507266524, 0.01553823795507762, 0.015266106106559828, 0.015017114770440192, 0.014796014809056983, 0.014590944410083614, 0.014398398609972291, 0.014195239445607018, 0.013978647109330273, 0.013766036302789912, 0.013556896906984486, 0.013352903234779159, 0.013186217580507635, 0.013021781457469392, 0.012864238046302606, 0.012724123543585226, 0.01259247941507749, 0.012461672187613468, 0.012328577870359455, 0.012193964876788393, 0.012070317827988874, 0.011960886149746598, 0.011859501836263782, 0.011762713912521115, 0.011661836687515546, 0.01156007391063278, 0.011464145066797006, 0.01138159340022065, 0.011279702693026687, 0.011187868817241422, 0.011076061400821433, 0.010995719807018174, 0.010920922282204175, 0.010844763089946866, 0.01077481257670432, 0.010699459354308912, 0.010626786050828679, 0.01056175981018262, 0.01050449383873487, 0.010442163638883303, 0.010388717777319286, 0.010336242905030216, 0.010286603033800596, 0.010236825189312852, 0.010190896376244483, 0.010144955011727958, 0.01009411911325947, 0.010044610731662455, 0.009994581126322623, 0.009941233270675602, 0.00989029605178976, 0.009843491532660724, 0.009795805596244397, 0.009760127073072021, 0.009718132523425814, 0.009679876739532623, 0.009637589121206507, 0.009603360141195293, 0.009564215844946352, 0.009521612648358733, 0.009489020046279169, 0.009453905805005674, 0.009419796611584782, 0.009376825020286252, 0.0093436230495377, 0.00930939112815129, 0.009273413598406296, 0.00923707528101594, 0.009206223671327531, 0.009173760034121164, 0.009143607053338857, 0.009115501174040449, 0.009085868118243584, 0.009057549313865093, 0.009031383140900861, 0.009003793807867736, 0.008984486395636763, 0.00896079682793543, 0.008932573011007885, 0.008914502452737047, 0.008890435737464225, 0.008869096533126277, 0.008846920990683994, 0.008827180727928279, 0.008807309326445611, 0.008788393361123801, 0.00876654829119449, 0.00874616764851645],
#     'ostatnia warstwa': [0.1792080939685341, 0.03883086941466451, 0.029641119681964236, 0.02535562814181632, 0.022514301993587976, 0.020714585959547165, 0.019294933619447528, 0.0181625421314235, 0.017298655058502508, 0.016640711489849167, 0.016114324503891216, 0.015663849143270736, 0.015263883254724098, 0.014927911209871959, 0.014629628877530776, 0.0143615933000209, 0.014097216222863368, 0.013856192495525404, 0.0136075318472041, 0.013389761744195513, 0.013190606908408281, 0.013005900362835466, 0.012836620251162821, 0.012677123446038097, 0.012533898271050153, 0.012404740695099966, 0.01227706788497407, 0.012158259424113511, 0.012041352619897984, 0.011934523479099846, 0.011827892323200973, 0.011727405433722679, 0.011630133058711049, 0.0115373432698584, 0.011452356592532825, 0.011373312177436232, 0.011298850449417214, 0.011232434710309425, 0.011161857415348343, 0.011090357946559834, 0.01103031811075963, 0.010957707741598828, 0.010896327750226777, 0.010835996351118373, 0.01077920983127918, 0.010722779855836624, 0.010672980811905385, 0.01061441042916543, 0.010560030674744491, 0.010513280585146997, 0.010467236168581775, 0.010418645974171692, 0.010379028740694764, 0.010334860738057937, 0.010292114241897913, 0.01025470147998112, 0.010216817863644305, 0.010174616554696172, 0.010132511764990233, 0.01009288299701412, 0.010054748702755294, 0.010016419688916575, 0.009980242583951632, 0.009941444602860791, 0.009906168364975813, 0.009868039308153234, 0.009828920515971125, 0.009794495534287278, 0.009762535227345048, 0.009728167637974535, 0.009697092867193264, 0.009666144594648766, 0.009635874148130911, 0.009605517458758114, 0.009583565535076376, 0.009556091931540977, 0.009527776822781612, 0.009501932650194295, 0.00947598233379495, 0.00945743861217198, 0.009431324437735367, 0.009408117679227116, 0.009391188652953473, 0.009368676087985145, 0.009345874218375861, 0.009321864122069087, 0.009304587855005129, 0.009281794765353384, 0.009260534082049196, 0.009241993710152633, 0.009221156066240771, 0.009204692122056877, 0.009180351123750288, 0.00916365072754929, 0.009146684935576032, 0.009123756872570797, 0.009106808910940296, 0.009090690156442198, 0.009072421664485351, 0.009051357228487756],
#     'uśredniony koszt':[0.12961670864294322, 0.040786616090692715, 0.040546745594600656, 0.04045119685176971, 0.04037061253828662, 0.04028389016954851, 0.04021822369977472, 0.04017549609195832, 0.04014535480337952, 0.04012169814561563, 0.04010312501715042, 0.04008712699641694, 0.04007402753226019, 0.04006271247185955, 0.04005362333961271, 0.040044557277719085, 0.04003681870573546, 0.04003194539900137, 0.040026780724057215, 0.0400194595515043, 0.0400157847645231, 0.04001227542129338, 0.040007152606819585, 0.04000500649005686, 0.040001517887925525, 0.03999695476553174, 0.03999448755952745, 0.0399921040349023, 0.03999210282421004, 0.03999392763007804, 0.03999369621768936, 0.039990412652848036, 0.03998870045653359, 0.03998475096434797, 0.03998136907709713, 0.03997793696129222, 0.03997532319757683, 0.039972463778767744, 0.03997058110137665, 0.03996927547546313, 0.03997013774793405, 0.03997195002037337, 0.03997155740761743, 0.03996946316148467, 0.03996723428084592, 0.03996574279795579, 0.03996483487035599, 0.03996237257974055, 0.03996106481750257, 0.03996035037738381, 0.039959067086411976, 0.039958820112238176, 0.039957951997662305, 0.03995781425892839, 0.03995794168618449, 0.03995652682266334, 0.039959314186315674, 0.039958531798939566, 0.03995842290437947, 0.03995931664726174, 0.03995840370997565, 0.039959071273434496, 0.03995868534495708, 0.03995784887854863, 0.039957997876612084, 0.03995847867588803, 0.039959770838947084, 0.03996036150935763, 0.039960263057424644, 0.039959481482415446, 0.039960363370376964, 0.03995977545518548, 0.03995796664627982, 0.03995737419343113, 0.03995745334907573, 0.03995720154730723, 0.03995603564525105, 0.03995611640774548, 0.039956909970450545, 0.03996243341404545, 0.03996421816897819, 0.03995838289083802, 0.0399616290775884, 0.03996862117163154, 0.03997116637949181, 0.039971608927070766, 0.03997602703302865, 0.03997828416441775, 0.03998526070480486, 0.03998547500383639, 0.03998169013924175, 0.039977642466092755, 0.03997428063330329, 0.03997338661132231, 0.039972420322286345, 0.03997345156111376, 0.039971736101241245, 0.0399689973473705, 0.039969524351843644, 0.0399692199212722]
# }
#
# plt.figure(figsize=(9,8))
# for key, data in results.items():
#     plt.plot([x for x in range(1, 101)], data, label=key)
#     plt.xlabel('epoki')
#     plt.ylabel('koszt')
#     plt.title('Training loss dla imputacji i metod losujących z 5 próbkami')
# plt.legend()
# plt.savefig('Training_loss_autoencoder_100_cnn_5.png')





import sys
from PIL import Image

for i in range(1000):
    images = [Image.open(x) for x in ['original_data_cnn/'+str(i)+'.png',
                                      'image_with_patch_cnn/'+str(i)+'.png',
                                      'result_cnn_3imputation_100_1_0.0/'+str(i)+'-imputation.png',
                                      'result_cnn_3first_layer_100_10_0.5/'+str(i)+'-first_layer.png',
                                      'result_cnn_3last_layer_100_10_1.5/'+str(i)+'-last_layer.png',
                                      'result_cnn_3first_layer_50_10_0.5/'+str(i)+'-first_layer.png']
              ]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]

    new_im.save('comparision/comparison-' + str(i) + '.png')


images = [Image.open(x) for x in ['comparision/comparison-1.png',
                                  'comparision/comparison-7.png',
                                  'comparision/comparison-12.png',
                                  'comparision/comparison-13.png',
                                  'comparision/comparison-16.png',
                                   'comparision/comparison-19.png',
                                  ]
          ]
widths, heights = zip(*(i.size for i in images))

max_width = max(widths)
total_height = sum(heights)

new_im = Image.new('RGB', (max_width, total_height))

y_offset = 0
for im in images:
    new_im.paste(im, (0, y_offset))
    y_offset += im.size[1]

new_im.save('comparision/comparison_all.png')

