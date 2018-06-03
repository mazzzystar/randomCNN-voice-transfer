import pandas as pd
import glob
from utils import *
from model import RandomCNN


data_path = 'VCTK-Corpus1/'

randomCNN = RandomCNN()
randomCNN.eval()
if cuda:
    randomCNN = randomCNN.cuda()


def process_vctk(_data_path, speaker_num=30, each_audio_num=15):
    # read label-info
    df = pd.read_table(_data_path + 'speaker-info.txt', usecols=['ID'],
                       index_col=False, delim_whitespace=True)

    # read file IDs
    file_ids = []
    for d in [_data_path + 'txt/p%d/' % uid for uid in df.ID.values[:speaker_num]]:
        file_ids.extend([f[-12:-4] for f in sorted(glob.glob(d + '*.txt')[:each_audio_num])])

    audio_lst = []
    for i, f in enumerate(file_ids):
        # wave file name
        wave_file = _data_path + 'wav48/%s/' % f[:4] + f + '.wav'
        fn = wave_file.split('/')[-1].split("_")[0]
        print(fn)
        # target_filename = 'asset/data/preprocess/mfcc/' + fn + '.npy'
        # if os.path.exists(target_filename):
        #     continue
        # print info
        print("VCTK corpus preprocessing (%d / %d) - '%s']" % (i, len(file_ids), wave_file))

        # load wave file
        spect, sr = wav2spectrum(wave_file)
        audio_lst.append((spect, fn))

        del spect
    return audio_lst


def compute_loss(a_C, a_G):
    """
    Compute the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_C, n_H, n_W)
    a_G -- tensor of dimension (1, n_C, n_H, n_W)

    Returns:
    J_content -- scalar that you compute using equation 1 above
    """
    n_H, n_W = a_G.shape

    # Reshape a_C and a_G to the (m * n_C, n_H * n_W)
    J_content = 1.0 / (n_H * n_W) * torch.sum((a_C - a_G) ** 2)

    return J_content

GAP_LEN = 15
TRAIN_LEN = 9
audio_lst = process_vctk(data_path)
print(len(audio_lst))

train_lst = []
test_lst = []

count = 0
for item in audio_lst:
    if count % GAP_LEN < TRAIN_LEN:
        train_lst.append(item)
    else:
        test_lst.append(item)
    count += 1
del audio_lst
print("Train len={}".format(len(train_lst)))
print("Test len={}".format(len(test_lst)))
for item in train_lst[:100]:
    print(item[-1])
for item in test_lst[:100]:
    print(item[-1])


def spect2gram(spect_lst):
    grams_lst = []
    for item in spect_lst:
        audio, no = item[0], item[1]
        audio = audio.T
        audio_delta = np.zeros(audio.shape)
        for i in range(audio.shape[0] - 1):
            audio_delta[i] = audio_delta[i+1] - audio_delta[i+1]

        audio = audio.T
        audio_delta = audio_delta.T
        audio_torch = torch.from_numpy(audio)[None, None, :, :]
        audio_delta_torch = torch.from_numpy(audio_delta)[None, None, :, :]
        audio_delta_var = Variable(audio_delta_torch, requires_grad=False).float()
        audio_var = Variable(audio_torch, requires_grad=False).float()
        if cuda:
            audio_var = audio_var.cuda()
            audio_delta_var = audio_delta_var.cuda()
        randomCNN_output = randomCNN(audio_var)
        gram = gram_over_time_axis(randomCNN_output)
        grams_lst.append((gram, no))
        del gram
        del randomCNN_output
        del audio_torch
        del audio_var
        del audio
    return grams_lst


train_grams = spect2gram(train_lst)
print("Train audio nums={}".format(len(train_grams)))
del train_lst

test_grams = spect2gram(test_lst)
print("Test audio nums={}".format(len(test_grams)))
del test_lst


def classifiy(new_gram, no):
    MIN_DIS = 100000
    MIN_NO = ""
    for item in train_grams:
        item_gram, item_no = item[0], item[1]
        dis = compute_loss(new_gram, item_gram)
        if dis.data[0] < MIN_DIS:
            MIN_DIS = dis.data[0]
            MIN_NO = item_no
        del item_gram
    return 1 if(MIN_NO == no) else 0

correct_count = 0
print("Begin to classify.")
for item in test_grams:
    gram, no = item[0], item[1]
    correct_count += classifiy(gram, no)
precise = float(correct_count) / len(test_grams)
print("test: {}/{}, precise={}".format(correct_count, len(test_grams), precise))






