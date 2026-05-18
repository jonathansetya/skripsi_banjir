def label_banjir(rain):

    if rain <= 1:
        return 0  # aman
    elif rain <= 3:
        return 1  # waspada
    else:
        return 2  # bahaya