def label_banjir(rain):

    if rain < 100:
        return 0  # aman

    elif rain < 300:
        return 1  # waspada

    else:
        return 2  # bahaya