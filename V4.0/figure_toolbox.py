import os
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def warn(message, category=UserWarning):
    """
    # Method of complain warning message
    - @message: Message of warning
    - @category: Category of warning, UserWarning by default
    """
    warnings.warn(f'Warning: {message}', category)


class Drawer(object):
    """
    # An easy-to-use figure collection
    The fig objects can be automatically stored and saved into single .pdf file.
    ## Example usage (D refers the instance):
      - Invite the new fig: ```D.fig(fig)```
      - Get the latest fig: ```fig = D.fig```
      - Clear the collection: ```D.clear()```
      - Save the figs into one single .pdf file: ```D.save('filename.pdf'[, (override boolean)])```
    """

    def __init__(self):
        """
        Startup with empty collection
        """
        # The latest fig
        self._fig = None
        # The list of figs,
        # it performes as a stack
        self.figures = []

    @property
    def fig(self):
        """
        # Get the latest fig
        """
        return self._fig

    @fig.setter
    def fig(self, f):
        """
        # Invite new fig and restore it
        - @f: The invited fig and it should be an instance of plt.figure, but we will **NOT** check it
        """
        self._fig = f
        self.figures.append(f)
        print(f'Got new figure, {len(self.figures)} in total.')

    def clear(self):
        """
        # Clear the collection
        The method is designed for clear the collection **MANUALLY** when there are too many figures,
        since it will slow down the system according to the complain of matplotlib library
        """
        count = 0
        while len(self.figures) > 0:
            fig = self.figures.pop()
            fig.clear()
            plt.close(fig)
            count += 1
        print(f'Figures has been cleared, {count} -> 0.')

    def save(self, filename, override=True):
        """
        # Draw figures into single .pdf file
        - @filename: The file of .pdf file, note that if it does not end with '.pdf', we will add it automatically,
                     it will **Return** 0 if success.
        - @override: If True is received (which is by default),
                     the .pdf file **WILL OVERRIDE EXISTING FILE** with the same name,
                     if it does so, a warning message will be complained.
                     if False is received, **EXISTING FILE** will be maintained,
                     and **Return** 1 since the saving process can not be properally operated.
        """

        if not filename.endswith('.pdf'):
            # Make sure filename endswith '.pdf'
            warn(
                f'Illegal filename: {filename}, its extend should be .pdf, fixing it.')
            filename = f'{filename}.pdf'

        if os.path.exists(filename):
            # Operation if file exists
            warn(f'{filename} exists.')
            if not override:
                # Stop here if file exists and not intend to override it
                warn(f'Not overriding permission, escaping at doing nothing.')
                return 1

        # Write figures into the .pdf file
        with PdfPages(filename, 'w') as pp:
            for f in self.figures:
                pp.savefig(f)
        print('Saved {} figures into {}.'.format(len(self.figures), filename))
        return 0
