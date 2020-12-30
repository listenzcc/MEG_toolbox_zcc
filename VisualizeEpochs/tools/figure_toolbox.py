import os
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class Drawer():
    """A figure collection, called drawer,
    the aim is to easily collect fig objects,
    and record them.
    """

    def __init__(self):
        # The latest fig object
        self._fig = None
        # The fig object collection
        self.figures = []

    @property
    def fig(self):
        """Link fig attr to self._fig """
        return self._fig

    @fig.setter
    def fig(self, f):
        """Update the latest fig object,
        and record it into self.figures.
        """
        self._fig = f
        self.figures.append(f)
        print(f'Got new figure, {len(self.figures)} in total.')

    def clear(self):
        count = 0
        while len(self.figures) > 0:
            fig = self.figures.pop()
            fig.clear()
            plt.close(fig)
            count += 1
        print(f'{count} figures has been cleared.')

    def save(self, filename, override=True):
        """Draw fig objects into .pdf file

        Arguments:
            filename {str} -- The filename of the .pdf file, a full path will work as the same

        Keyword Arguments:
            override {bool} -- Whether override the .pdf file if it already exists

        Returns:
            Flag of success, 0 means success, 1 means fail.
        """

        def warn(content):
            """Local warning method"""
            warnings.warn(f'Warning: {content}', RuntimeWarning)

        # Regulation filename
        if not filename.endswith('.pdf'):
            warn(
                f'Illegal filename: {filename}, its extend should be .pdf, fixing it.')
            filename = f'{filename}.pdf'

        # Check file exists
        if os.path.exists(filename):
            warn(f'{filename} exists.')
            if not override:
                warn(f'Not overriding permission, escaping at doing nothing.')
                return 1

        # Write figures into the .pdf file
        with PdfPages(filename, 'w') as pp:
            for f in self.figures:
                pp.savefig(f)
        print('Saved {} figures into {}.'.format(len(self.figures), filename))
        return 0
