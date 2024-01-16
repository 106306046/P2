import numpy as np

def MI_pred(raw,model,binary_class_dict,band_dict,CSPs, sfreq):
    sample_per_step = int(sfreq / 2)
    data_of_combination = {}

    for binary_class in binary_class_dict.values():
        class1, class2 = binary_class
        Bl, Bh = band_dict[binary_class]
        epochs_fil = raw.copy().filter(Bl, Bh, method='iir', verbose=False)
        data_fil = epochs_fil.get_data(copy = True)

        # 原本的算法
        data_csp = CSPs[(class1, class2)].transform(data_fil)
        section1 = np.log(np.var(data_csp[:, :, :sfreq], axis=2, keepdims=True))
        section2 = np.log(
            np.var(data_csp[:, :, sample_per_step * 1:sample_per_step * 1 + sfreq], axis=2, keepdims=True))
        section3 = np.log(
            np.var(data_csp[:, :, sample_per_step * 2:sample_per_step * 2 + sfreq], axis=2, keepdims=True))
        data_of_combination[binary_class] = np.concatenate((section1, section2, section3), axis=2)

    merged_arrays = []

    for binary_class in data_of_combination:
        array = data_of_combination[binary_class]
        merged_arrays.append(array)

    data_merged = np.concatenate(merged_arrays, axis=1)

    X_shape = data_merged.shape
    X_test = data_merged.reshape((X_shape[0], X_shape[1], X_shape[2], 1, 1))

    # print(model.predict(X_test))
    pred = np.argmax(model.predict(X_test), axis=-1)[0]
    return pred