# To make figure operation easier

import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class Drawer():
    """A figure collection, called drawer,
    the aim is to easily collect fig objects,
    and save them into .pdf file.
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
        print(f'New figure added, {self.figures.__len__()} in total.')

    def clear_figures(self):
        for fig in self.figures:
            plt.close(fig)
        print(f'Cleared all figures, {len(self.figures)} in total.')
        self.figures = []

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
            pre = '------> '
            Warning(f'{pre}content')

        # Regulation filename
        if not filename.endswith('.pdf'):
            warn(f'Illegal filename: {filename}, its extend should be .pdf')
            filename = f'{filename}.pdf'

        # Check file exists
        if os.path.exists(filename):
            warn(f'{filename} exists.')
            if not override:
                warn(f'Not overriding permission, do nothing and escaping.')
                return 1

        # Write figures into the .pdf file
        with PdfPages(filename, 'w') as pp:
            for f in self.figures:
                pp.savefig(f)

        print('Saved {} figures into {}.'.format(len(self.figures), filename))
        return 0
