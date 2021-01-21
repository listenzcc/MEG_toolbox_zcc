# General functions

import warnings


def warn(message, category=UserWarning):
    """
    # Method of complain warning message
    - @message: Message of warning
    - @category: Category of warning, UserWarning by default
    """
    warnings.warn(f'Warning: {message}', category)
