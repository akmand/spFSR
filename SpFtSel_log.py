
# spFtSel: Feature Selection and Ranking via SPSA
# V. Aksakalli & Z. D. Yenice
# GPL-3.0, 2019
# Please refer to below for more information:
# https://arxiv.org/abs/1804.05589

import logging


class SpFtSel_log:

    # create logger
    logger = logging.getLogger('spFtSel')
    logger.setLevel(logging.INFO)

    # create console handler and set level to info
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter(
        fmt='{name}-{levelname}: {message}',
        style='{',
    )

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

