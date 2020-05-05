import numpy as np


class gait:
    def __int__(self, n_segment, offsets, durations, name):
        self.name = name

        self.mpc_table = [None] * n_segment * 4
        self.n_iteration = n_segment

        self.offsets = offsets
        self.durations = durations
        self.offsetsFloat = np.array(offsets) / n_segment
        self.durations = np.array(durations) / n_segment

        self.stance = durations[0]
        self.swing = n_segment - durations[0]

        self.phase = 0.0

    def get_contact_state(self):
        progress = self.phase

        cyclic = progress < 0  # cyclic is an array that has one on place where progress is less than zero
        progress = progress + cyclic

